# core/data_gathering.py

import os
import time
import json
import numpy as np
import pandas as pd
import requests
import traceback
from typing import List, Union, Tuple
from tqdm import tqdm
from requests import  Session
from typing import List, Union, Tuple, Dict
import ast
import re

import  socket, random
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

session = Session()




class OpenAlexDataGatherer:
    """Gather data from OpenAlex and persist articles, authors, and citations via HTTP with rate-limit enforcement."""

    BASE_URL = "https://api.openalex.org/works"
    RATE_LIMIT = 9999999          # max requests per window
    WINDOW_SECONDS = 24 * 3600  # 24h window

    def __init__(self, path: str, folder: str, email: str):
        """
        Initialize gatherer with storage path, subfolder, and contact email.
        """
        self.path = path
        self.folder = folder
        self.email = email
        self.test_mode=False
         
        self.data_lenght=50000 #to init empty numpy array for faster storing data to the pandas
        self.data_index=0  # track indexing the data to the numpoy array
        os.makedirs(os.path.join(self.path, self.folder), exist_ok=True)
        # prepare request log file
        self._log_file = os.path.join(self.path, self.folder, ".request_log.json")
        if not os.path.exists(self._log_file):
            with open(self._log_file, 'w') as f:
                json.dump([], f)

    def _enforce_rate_limit(self):
        """Sleep if exceeding RATE_LIMIT requests in WINDOW_SECONDS."""
        now = time.time()
        with open(self._log_file, 'r+') as f:
            try:
                timestamps = json.load(f)
            except json.JSONDecodeError:
                timestamps = []
            window_start = now - self.WINDOW_SECONDS
            timestamps = [ts for ts in timestamps if ts >= window_start]
            if len(timestamps) >= self.RATE_LIMIT:
                wait = self.WINDOW_SECONDS - (now - min(timestamps))
                time.sleep(wait)
                now = time.time()
                timestamps = [ts for ts in timestamps if ts >= now - self.WINDOW_SECONDS]
            timestamps.append(now)
            f.seek(0)
            json.dump(timestamps, f)
            f.truncate()

    def time_str2date(self, publication_date: str) -> Tuple[int, str]:
        """Parse year and return (year, full_date) from 'YYYY-MM-DD'."""
        year = None
        if publication_date:
            try:
                year = int(publication_date.split('-')[0])
            except ValueError:
                year = None
        return year, publication_date

    def make_db_articles(
        self,
        terms: Union[str, List[str]],
        filter_by: str = 'abstract',
        max_per_term: int = 200,
        max_download: int = 1000000,
        data_length: int = 50000    
        ) -> pd.DataFrame:
        """
        Fetch articles by 'abstract', 'institution', or 'author' keywords.
        Returns DataFrame and saves article & author JSON files.
        
        data_length - 
        """
        sess = openalex_session(trust_env=False)
        filter_map = {
            'abstract': 'abstract.search',
            'institution': 'institutions.search',
            'author': 'authorships.author.display_name.search',
            "concept": "concepts.id"
        }
        self.data_length=data_length
        if filter_by not in filter_map:
            raise ValueError(f"filter_by must be one of {list(filter_map)}")
        filter_key = filter_map[filter_by]
        if isinstance(terms, str):
            terms = [terms]
        
        art_file = os.path.join(self.path, self.folder, "pd_articles.json")
        auth_file = os.path.join(self.path, self.folder, "pd_authors.json")
        self.art_colums=[
            'Paper ID', 'Title', 'Abstract', 'Year', 'Date of publish',
            'No. of citations', 'Journal', 'Journal ID', 'ISSN', 'Type',
            'Publisher', 'Doi'
            ]
        self.aut_colums=[
            'Paper ID', 'Title', 'Author', 'Author ID', 'Author ORCID',
            'Institution', 'Institution ID', 'Country', 'Authors place', 'Year',
            'No. of citations', 'Journal', 'ISSN', 'Type', 'Publisher', 'Latitude',
            'Longitude'
            ]
        try:
            articles_pd = pd.read_json(art_file)
            print(f"Loaded 'articles_pd' from {art_file}")
        except Exception:
            articles_pd = pd.DataFrame(columns=self.art_colums)
            print(f"Initialized empty 'articles_pd' DataFrame instead of {art_file}")
        try:
            authors_pd = pd.read_json(auth_file)
            print(f"Loaded 'authors_pd' from {auth_file}")
        except Exception:
            authors_pd = pd.DataFrame(columns=self.aut_colums)
            print(f"Initialized empty 'authors_pd' DataFrame instead of {auth_file}")
            
            
        
        self.existing_ids  = set(articles_pd['Paper ID'].tolist())
        self.existing_idsa = set([])
        authors_ids  = np.asarray(authors_pd["Paper ID"].values)
        authors_pids = np.asarray(authors_pd["Author ID"].values)
        if len(authors_pd)>0:
            for idx in range(len(authors_ids)):
                pida=authors_ids[idx]+"_"+authors_pids[idx]
                self.existing_idsa.add(pida)
        
        

        self.data_paper_index=0
        self.data_author_index=0
        self.data_papers = np.empty((self.data_length, len(self.art_colums)), dtype=object)
        self.data_authors = np.empty((self.data_length, len(self.aut_colums)), dtype=object)
        papers_index=0
        retrieved = 0
        page_no = 0
        with tqdm(total=self.data_length, initial=self.data_author_index, desc="Papers", unit="paper", dynamic_ncols=True) as pbar:
            for term in terms:
                cursor = '*'
                while retrieved < max_download and cursor:
                    self._enforce_rate_limit()
                    self.params = {
                        'filter': f'{filter_key}:{term}',
                        'per-page': min(200, max_per_term),
                        'cursor': cursor,
                        'mailto': self.email
                    }
                    headers = { 'User-Agent': 'MyOpenAlexClient/0.1 (mailto:jaromir.klarak@savba.sk)'}
                    #resp = requests.get(self.BASE_URL, params=self.params,headers=headers)
                    resp = robust_get(sess, self.BASE_URL, params=self.params,headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    cursor = data.get('meta', {}).get('next_cursor')
                    page_no += 1
                    self.data=data
                    for idx, paper in enumerate(data.get('results', [])):
                        try:
                            
                            pid = paper['id'].split('/')[-1]
                            
                            if pid not in self.existing_ids: 
                                pbar.set_description(f"Page: {page_no} / Papers: {papers_index}")
                                papers_index+=1
                                self.existing_ids.add(pid)
                                year, date_str = self.time_str2date(paper.get('publication_date',''))
                                'Paper ID', 'Title', 'Abstract', 'Year', 'Date of publish',
                                'No. of citations', 'Journal', 'Journal ID', 'ISSN', 'Type',
                                'Publisher', 'Doi'
                                if paper.get('abstract_inverted_index') != None:
                                    abstract=self.abstract_from_inverted(paper.get('abstract_inverted_index'))
                                else:
                                    abstract=None
                                
                                src = ((paper.get('primary_location') or {}).get('source') or {})
                                row = np.asarray([
                                    pid,                                                                                # 1 'Paper ID'
                                    self.remove_non_ascii(paper.get('title','')),                                            # 2 'Title'
                                    abstract,                                                                           # 3 'Abstract'
                                    year,                                                                               # 4 'Year'
                                    date_str,                                                                           # 5 'Date of publish'
                                    paper.get('cited_by_count'),                                                        # 6 'No. of citations'
                                    src.get('display_name'),                                                            # 7 'Journal'
                                    src.get('id','').split('/')[-1],                                                    # 8 'Journal ID'
                                    src.get('issn_l'),                                                                  # 9 'ISSN'
                                    paper.get('type'),                                                                  # 10 'Type'
                                    src.get('host_organization_name'),                                                  # 11 'Publisher'
                                    paper.get('ids',{}).get('doi')                                                      # 12 'Doi'
                                    ])    
                                
                                self.data_papers[self.data_paper_index,:]=row
                                
                                self.data_paper_index+=1
                                retrieved+=1
                             
                            src = ((paper.get('primary_location') or {}).get('source') or {})    
                            for auth in paper.get('authorships', []):
                                pida=pid+"_"+str(auth.get('author',{}).get('id','').split('/')[-1])
                                if pida not in self.existing_idsa:
                                    try:
                                        insts = auth.get('institutions') or []    # handles None or missing → []
                                        inst  = insts[0] if insts else {}         # {} when no institutions
                                        geo   = (inst.get('geo') or {})           # {} when no geo
                                        
                                        'Paper ID', 'Title', 'Author', 'Author ID', 'Author ORCID',
                                        'Institution', 'Institution ID', 'Country', 'Authors place', 'Year',
                                        'No. of citations', 'Journal', 'ISSN', 'Type', 'Publisher', 'Latitude',
                                        'Longitude'
                                        self.data_authors[self.data_author_index,:]=np.asarray([
                                            pid,                                                                        # 0 'Paper ID '
                                            self.remove_non_ascii(paper.get('title','')),                                    # 1 'Title': 
                                            auth.get('author',{}).get('display_name'),                                  # 2 'Author': 
                                            auth.get('author',{}).get('id','').split('/')[-1],                          # 3 'Author ID':
                                            auth.get('author',{}).get('orcid'),                                         # 4 'Author ORCID':
                                            inst.get('display_name'),                                                   # 5 'Institution': 
                                            inst.get('id','').split('/')[-1],                                           # 6 'Institution ID':
                                            inst.get('country_code'),                                                   # 7 'Country': 
                                            auth.get('author_position'),                                                # 8 'Authors place': 
                                            year,                                                                       # 9 YEAR
                                            paper.get('cited_by_count'),                                                # 10 'No. of citations'
                                            src.get('display_name'),                                                    # 11 'Journal'
                                            src.get('issn_l'),                                                          # 12 'ISSN'
                                            paper.get('type'),                                                          # 13 'Type'
                                            src.get('host_organization_name'),                                          # 14 'Publisher
                                            geo.get('latitude'),                                                        # 15 'Latitude': 
                                            geo.get('longitude')                                                        # 16 'Longitude': 
                                            ])
                                        self.data_author_index+=1
                                        pbar.update(1)
                                        self.existing_idsa.add(pida)
                                    except:
                                        pass
            
                            if self.data_author_index>49500:

                                # Number of valid rows already written into the buffers
                                n_papers  = int(self.data_paper_index)
                                n_authors = int(self.data_author_index)
                                
                                # Slice only the filled region (no need for None-masks)
                                papers_chunk  = self.data_papers[:n_papers, :]
                                authors_chunk = self.data_authors[:n_authors, :]
                                
                                # Turn chunks into DataFrames with the right columns
                                papers_df  = pd.DataFrame(papers_chunk,  columns=self.art_colums)
                                authors_df = pd.DataFrame(authors_chunk, columns=self.aut_colums)
                                
                                # Append to your main DataFrames
                                articles_pd = pd.concat([articles_pd, papers_df], ignore_index=True)
                                authors_pd  = pd.concat([authors_pd,  authors_df], ignore_index=True)
                                
                                articles_pd.to_json(art_file, orient='records', force_ascii=False)
                                authors_pd.to_json(auth_file, orient='records', force_ascii=False)
                                
                                self.data_paper_index=0
                                self.data_author_index=0
                                print(f"Stored new data with {n_papers} Papers and {n_authors} Authors")
                                self.data_papers = np.empty((self.data_length, len(self.art_colums)), dtype=object)
                                self.data_authors = np.empty((self.data_length, len(self.aut_colums)), dtype=object) 
                                pbar.reset(total=self.data_length)  # resets total & sets counter to 0

                                
                        except:
                            pass   
        articles_pd.to_json(art_file, orient='records', force_ascii=False)
        authors_pd.to_json(auth_file, orient='records', force_ascii=False)        
        return articles_pd

    def get_citing_papers(self, paper_id: str) -> List[str]:
        """Return list of OpenAlex work IDs citing the given paper, saving every page."""

        paper_id=str(paper_id)
        if not isinstance(paper_id, str):
            raise AttributeError(f"paper_id must be a string but is {type(paper_id)}")
        citing, cursor = [], '*'
        cite_file = os.path.join(self.path, self.folder, f"citing_{paper_id}.json")
        while cursor:
            self._enforce_rate_limit()
            params = {'filter': f'cites:{paper_id}', 'per-page':200, 'cursor':cursor, 'mailto':self.email}
            resp = requests.get(self.BASE_URL, params=params); resp.raise_for_status()
            data = resp.json(); cursor = data.get('meta', {}).get('next_cursor')
            for work in data.get('results',[]): citing.append(work['id'].split('/')[-1])
            if getattr(self, "test_mode", False):
                with open(cite_file, 'w') as f: json.dump(citing, f, indent=2)
        return citing

    def get_articles_citations(self, articles_df: pd.DataFrame, save_every: int = 100, citations_file: str = 'pd_articles_citations.json') -> pd.DataFrame:
        
        """
        For each article in articles_df, fetch citing papers and record counts and years.
        Skips already-processed articles and saves periodically.
        """
        
        cit_path = os.path.join(self.path, self.folder, citations_file)
        try:
            citations_df = pd.read_json(cit_path)
            print(f"Loaded citations from {cit_path}")
        except Exception:
            citations_df = pd.DataFrame(index=range(len(articles_df)),columns=['Paper ID','Citing papers','Num citations','Year'])
            print(f"Initialized empty citations DataFrame: {cit_path}")

        existing = np.unique(np.asarray(citations_df['Paper ID'].tolist()),return_counts="True")[0]
        papers_id=articles_df["Paper ID"].values
        years=articles_df['Year'].values
        for idx, pid in tqdm(enumerate(papers_id), total=len(articles_df)):
            
            if pid not in existing: 
                try:
                    cites = self.get_citing_papers(pid)
                    citations_df.at[idx, 'Paper ID'] = pid
                    citations_df.at[idx, 'Citing papers'] = cites  # Storing the list
                    citations_df.at[idx, 'Num citations'] = len(cites)
                    citations_df.at[idx, 'Year'] = years[idx]
                    
                except Exception as e:
                    print(f"ERROR fetching {pid}: {e}")
                    # if you want the full stack trace:
                    traceback.print_exc()

                    #continue
                if len(citations_df) % save_every == 0:
                    citations_df.to_json(cit_path, orient='records', force_ascii=False)
            
        
        citations_df.to_json(cit_path, orient='records', force_ascii=False)
        return citations_df

    def get_citing_insides_and_edges(self, source:str="pd_articles_citations.json") -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Process stored citations to filter only internal citations and construct edges.
        Returns a DataFrame of internal citation counts and an edges array.
        """
        # Load previously fetched citations
        key=source.split(".")[0]
        cite_path = os.path.join(self.path, self.folder, source)
        article_citing = pd.read_json(cite_path)

        # Prepare output structures
        processed_df = pd.DataFrame(columns=[
            'Paper ID', 'Citing papers', 'Num citations', 'Year'
        ])
        all_ids = article_citing['Paper ID'].tolist()

        edges = np.empty((0, 2), dtype=object)
        start_time = time.time()

        for idx, row in tqdm(enumerate(article_citing.itertuples()),total=len(article_citing), desc= key):
            row=list(row)
            pid = row[1]   # Paper ID
            year = row[4]  # Year
            citing_list = np.asarray(row[2])  # Citing papers list
            internal = citing_list[np.isin(citing_list, all_ids)]

            row_data = {
                        "Paper ID": pid,
                        'Citing papers': [internal],
                        "Num citations": len(internal),
                        'Year': np.float32(year)
                        }


            processed_df.loc[len(processed_df)] = row_data


            # Build edge list
            if len(internal) > 0:
                src = np.full(len(internal), pid, dtype=object)
                edges = np.vstack((edges, np.vstack((src, internal)).T))

            # Periodic save
            if idx % 100 == 0:
                elapsed = time.time() - start_time
                processed_df.to_json(
                    os.path.join(self.path, self.folder, key+"_processed.json"),
                    orient='records', force_ascii=False
                )
                pd.DataFrame(edges).to_json(
                    os.path.join(self.path, self.folder, "edges_"+key+".json"),
                    orient='records', force_ascii=False
                )
                start_time = time.time()

        # Final save
        processed_df.to_json(
            os.path.join(self.path, self.folder, key+"_processed.json"),
            orient='records', force_ascii=False
        )
        pd.DataFrame(edges).to_json(
            os.path.join(self.path, self.folder, "edges_"+key+".json"),
            orient='records', force_ascii=False
        )
        return processed_df, edges

    def get_articles_references(
        self,
        articles_df: pd.DataFrame,
        save_every: int = 100,
        references_file: str = 'pd_articles_references.json'
        ) -> pd.DataFrame:
        """
        For each article in articles_df, fetch referenced works (papers this article cites) and record counts and years.
        Skips already-processed articles and saves periodically.
        """
        ref_path = os.path.join(self.path, self.folder, references_file)
        try:
            references_df = pd.read_json(ref_path)
            print(f"Loaded references from {ref_path}")
        except Exception:
            references_df = pd.DataFrame(columns=['Paper ID','References','Num references','Year'])
            print("Initialized empty references DataFrame")
        existing = np.unique(np.asarray(references_df['Paper ID'].tolist()),return_counts="True")[0]
        for idx, row in tqdm(articles_df.iterrows(), total=len(articles_df), desc='References'):
            pid = row['Paper ID']
            if pid not in existing:
                try:
                    # Fetch single work details
                    self._enforce_rate_limit()
                    url = f"{self.BASE_URL}/{pid}"
                    resp = requests.get(url, params={'mailto': self.email})
                    resp.raise_for_status()
                    work = resp.json()
                    refs = [r.split('/')[-1] for r in work.get('referenced_works', [])]
                    
        
                    references_df.loc[idx,['Paper ID','References','Num references','Year']]=pd.Series({'Paper ID':pid,
                         'References':refs,
                         'Num references':len(refs),
                         'Year':row.get('Year')})
                except Exception as err:
                    print(f"Wrong in {idx}: {pid} with {err}")
                    return {"pid":pid,"work":work, "refs":refs}
                    
    

                if len(references_df) % save_every == 0:
                    references_df.to_json(ref_path, orient='records', force_ascii=False)
            else:
                print(f"{idx}:{pid} already stored")
        # Final save
        references_df.to_json(ref_path, orient='records', force_ascii=False)
        return references_df

    def get_authors_by_name(self, name: str, max_results: int = 100) -> pd.DataFrame:
        """
        Query OpenAlex for authors whose display name contains the given substring.

        Parameters:
        - name: substring to search for in author display names
        - max_results: maximum number of author records to retrieve

        Returns:
        - DataFrame of matching author records
        """
        author_url = "https://api.openalex.org/authors"
        authors = []
        retrieved = 0
        cursor = '*'
        data_all=list()
        while retrieved < max_results and cursor:
            self._enforce_rate_limit()
            params = {
                'filter': f'display_name.search:{name}',
                'per-page': min(200, max_results - retrieved),
                'cursor': cursor,
                'mailto': self.email
            }
            resp = requests.get(author_url, params=params)
            resp.raise_for_status()
            data = resp.json()
            cursor = data.get('meta', {}).get('next_cursor')
            data_all.append(data)
            for auth in data.get('results', []):
                if retrieved >= max_results:
                    break
                retrieved += 1
                try:
                    last_known_institution = auth.get('last_known_institutions', {})

                    # If 'last_known_institution' is empty or None, handle gracefully
                    affiliations = [inst.get('display_name') for inst in last_known_institution] if isinstance(last_known_institution, list) else []
                    if isinstance(last_known_institution, list):
                        # Extract country_code from each dictionary in the list
                        country_codes = [inst.get('country_code') for inst in last_known_institution if isinstance(inst, dict)] #for this case is used for the future maintaining code, this store country codes in way of list, now it is complicating store list to the dictionary
                        country_codes_str = ', '.join([str(code) for code in country_codes if code is not None])
                    else:
                        country_codes_str=None
                        country_codes = None    
                                        
                    record = {
                        'Author ID': auth.get('id', '').split('/')[-1],
                        'Display Name': auth.get('display_name'),
                        'ORCID': auth.get('orcid'),
                        'Works Count': auth.get('works_count'),
                        'Citation Count': auth.get('cited_by_count'),
                        'Affiliations': affiliations,
                        'Country': country_codes_str
                    }
                    authors.append(record)
                except Exception as e:
                    print(f"Error with get authors records for: {name} in {e}")
                    if self.test_mode==True:
                        return auth,data_all
                    else:
                        return auth


        df_authors = pd.DataFrame(authors)
        if self.test_mode is True:
            return df_authors,data_all
        else:
            return df_authors


    def get_papers_by_author_id(self, author_id: str,max_results: int = 200) -> pd.DataFrame:
        """
        Fetch all works (papers) associated with a given OpenAlex author ID.

        Parameters:
        - author_id: OpenAlex author ID (without URL prefix)
        - max_results: maximum number of works to retrieve

        Returns:
        - DataFrame of works by the author
        """
        works = []
        retrieved = 0
        cursor = '*'
        while retrieved < max_results and cursor:
            self._enforce_rate_limit()
            params = {
                'filter': f'author.id:{author_id}',
                'per-page': min(200, max_results - retrieved),
                'cursor': cursor,
                'mailto': self.email
            }
            resp = requests.get(self.BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            cursor = data.get('meta', {}).get('next_cursor')
            for work in data.get('results', []):
                if retrieved >= max_results:
                    break
                retrieved += 1
                works.append({
                    'Work ID': work.get('id', '').split('/')[-1],
                    'Title': remove_non_ascii(work.get('title', '')),
                    'Publication Year': work.get('publication_year'),
                    'DOI': work.get('ids', {}).get('doi'),
                    'Cited By Count': work.get('cited_by_count'),
                    'Type': work.get('type')
                })
        df_works = pd.DataFrame(works)
        return df_works
    
    def remove_non_ascii(self,text):
        if not text:          # handles None and empty strings
            return ""
        if not isinstance(text, str):
            text = str(text)
        return "".join(c for c in text if ord(c) < 128)
    
    def abstract_from_inverted(self,
        inverted: Union[Dict[str, List[int]], str]
        ) -> str:
        """
        Convert OpenAlex's abstract_inverted_index to a plaintext abstract.
    
        Args:
            inverted: either the dict {word: [positions,...]} or a string
                      containing that Python literal.
    
        Returns:
            The abstract as a plain-text string ('' if not available).
        """
        # Parse if we were given a string literal
        if isinstance(inverted, str):
            inverted = ast.literal_eval(inverted)
    
        if not inverted:
            return ""
    
        # Place each word at its position, then join
        max_pos = max(p for positions in inverted.values() for p in positions)
        words = [""] * (max_pos + 1)
        for word, positions in inverted.items():
            for p in positions:
                words[p] = word
    
        text = " ".join(words).strip()
        # Tidy any accidental extra spaces (rare)
        text = re.sub(r"\s{2,}", " ", text)
        return text
    



def openalex_session(trust_env=False):
    s = requests.Session()
    s.trust_env = trust_env              # ignore env proxies unless you need them
    retry = Retry(
        total=0,                         # we'll handle backoff ourselves
        connect=0, read=0
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50))
    return s

def robust_get(sess, url, params, headers, max_tries=10, base_sleep=1.5):
    """Retry with exponential backoff; detect DNS outages and keep trying."""
    for attempt in range(1, max_tries + 1):
        try:
            r = sess.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            # Check if DNS is the culprit
            dns_ok = True
            try:
                socket.gethostbyname("api.openalex.org")
            except Exception:
                dns_ok = False

            wait = base_sleep * (2 ** (attempt - 1)) * (1 + 0.25 * random.random())
            msg = f"[retry {attempt}/{max_tries}] {'DNS fail' if not dns_ok else 'net/HTTP error'}: {e!r}; sleep {wait:.1f}s"
            tqdm.write(msg)
            time.sleep(wait)
    # If we get here, bubble up the last error
    raise







