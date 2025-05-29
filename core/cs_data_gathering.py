# core/data_gathering.py

import os
import time
import json
import numpy as np
import pandas as pd
import requests

from typing import List, Union, Tuple
from tqdm import tqdm


def remove_non_ascii(text: str) -> str:
    """Utility to remove non-ASCII characters from text."""
    return ''.join(c for c in text if ord(c) < 128)


class OpenAlexDataGatherer:
    """Gather data from OpenAlex and persist articles, authors, and citations via HTTP with rate-limit enforcement."""

    BASE_URL = "https://api.openalex.org/works"
    RATE_LIMIT = 99999          # max requests per window
    WINDOW_SECONDS = 24 * 3600  # 24h window

    def __init__(self, path: str, folder: str, email: str):
        """
        Initialize gatherer with storage path, subfolder, and contact email.
        """
        self.path = path
        self.folder = folder
        self.email = email
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
        max_per_term: int = 200
    ) -> pd.DataFrame:
        """
        Fetch articles by 'abstract', 'institution', or 'author' keywords.
        Returns DataFrame and saves article & author JSON files.
        """
        filter_map = {
            'abstract': 'abstract.search',
            'institution': 'institutions.search',
            'author': 'authorships.author.display_name.search'
        }
        if filter_by not in filter_map:
            raise ValueError(f"filter_by must be one of {list(filter_map)}")
        filter_key = filter_map[filter_by]
        if isinstance(terms, str):
            terms = [terms]

        art_file = os.path.join(self.path, self.folder, "pd_articles.json")
        auth_file = os.path.join(self.path, self.folder, "pd_authors.json")
        try:
            articles_pd = pd.read_json(art_file)
        except Exception:
            articles_pd = pd.DataFrame(columns=[
                'Paper ID','Title','Abstract','Year','Date of publish',
                'No. of citations','Journal','Journal ID','ISSN','Type','Publisher','Doi'
            ])
        try:
            authors_pd = pd.read_json(auth_file)
        except Exception:
            authors_pd = pd.DataFrame(columns=[
                'Paper ID','Title','Author','Author ID','Author ORCID',
                'Institution','Institution ID','Country','Authors place','Latitude','Longitude'
            ])

        existing_ids = set(articles_pd['Paper ID'].tolist())
        for term in terms:
            retrieved, cursor = 0, '*'
            while retrieved < max_per_term and cursor:
                self._enforce_rate_limit()
                params = {
                    'filter': f'{filter_key}:{term}',
                    'per-page': min(200, max_per_term - retrieved),
                    'cursor': cursor,
                    'mailto': self.email
                }
                resp = requests.get(self.BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
                cursor = data.get('meta', {}).get('next_cursor')
                for paper in data.get('results', []):
                    if retrieved >= max_per_term:
                        break
                    pid = paper['id'].split('/')[-1]
                    if pid in existing_ids:
                        continue
                    retrieved += 1
                    existing_ids.add(pid)
                    year, date_str = self.time_str2date(paper.get('publication_date',''))
                    rec = {
                        'Paper ID': pid,
                        'Title': remove_non_ascii(paper.get('title','')),
                        'Abstract': paper.get('abstract_inverted_index'),
                        'Year': year,
                        'Date of publish': date_str,
                        'No. of citations': paper.get('cited_by_count'),
                        'Journal': paper.get('primary_source',{}).get('display_name'),
                        'Journal ID': paper.get('primary_source',{}).get('id','').split('/')[-1],
                        'ISSN': paper.get('primary_source',{}).get('issn_l'),
                        'Type': paper.get('type'),
                        'Publisher': paper.get('host_venue',{}).get('publisher_name'),
                        'Doi': paper.get('ids',{}).get('doi')
                    }
                    articles_pd = pd.concat([articles_pd, pd.DataFrame(rec,index=[0])], ignore_index=True)
                    for auth in paper.get('authorships', []):
                        inst = auth.get('institutions',[{}])[0]
                        geo = inst.get('geo', {})
                        auth_rec = {
                            'Paper ID': pid,
                            'Title': remove_non_ascii(paper.get('title','')),
                            'Author': auth.get('author',{}).get('display_name'),
                            'Author ID': auth.get('author',{}).get('id','').split('/')[-1],
                            'Author ORCID': auth.get('author',{}).get('orcid'),
                            'Institution': inst.get('display_name'),
                            'Institution ID': inst.get('id','').split('/')[-1],
                            'Country': inst.get('country_code'),
                            'Authors place': auth.get('author_position'),
                            'Latitude': geo.get('latitude'),
                            'Longitude': geo.get('longitude')
                        }
                        authors_pd = pd.concat([authors_pd, pd.DataFrame(auth_rec,index=[0])], ignore_index=True)
                articles_pd.to_json(art_file, orient='records', force_ascii=False)
                authors_pd.to_json(auth_file, orient='records', force_ascii=False)
        return articles_pd

    def get_citing_papers(self, paper_id: str) -> List[str]:
        """Return list of OpenAlex work IDs citing the given paper, saving every page."""
        if not isinstance(paper_id, str):
            raise AttributeError("paper_id must be a string")
        citing, cursor = [], '*'
        cite_file = os.path.join(self.path, self.folder, f"citing_{paper_id}.json")
        while cursor:
            self._enforce_rate_limit()
            params = {'filter': f'cites:{paper_id}', 'per-page':200, 'cursor':cursor, 'mailto':self.email}
            resp = requests.get(self.BASE_URL, params=params); resp.raise_for_status()
            data = resp.json(); cursor = data.get('meta', {}).get('next_cursor')
            for work in data.get('results',[]): citing.append(work['id'].split('/')[-1])
            with open(cite_file, 'w') as f: json.dump(citing, f, indent=2)
        return citing

    def get_articles_citations(
        self,
        articles_df: pd.DataFrame,
        save_every: int = 100,
        citations_file: str = 'pd_articles_citations.json'
    ) -> pd.DataFrame:
        """
        For each article in articles_df, fetch citing papers and record counts and years.
        Skips already-processed articles and saves periodically.
        """
        cit_path = os.path.join(self.path, self.folder, citations_file)
        try:
            citations_df = pd.read_json(cit_path)
            print(f"Loaded citations from {cit_path}")
        except Exception:
            citations_df = pd.DataFrame(columns=['Paper ID','Citing papers','Num citations','Year'])
            print(f"Initialized empty citations DataFrame")
        existing = set(citations_df['Paper ID'].tolist())
        for idx, row in tqdm(articles_df.iterrows(), total=len(articles_df), desc='Citations'):
            pid = row['Paper ID']
            if pid in existing: continue
            try:
                cites = self.get_citing_papers(pid)
                citations_df = citations_df.append({
                    'Paper ID': pid,
                    'Citing papers': cites,
                    'Num citations': len(cites),
                    'Year': row.get('Year')
                }, ignore_index=True)
                existing.add(pid)
            except Exception:
                continue
            if len(citations_df) % save_every == 0:
                citations_df.to_json(cit_path, orient='records', force_ascii=False)
        citations_df.to_json(cit_path, orient='records', force_ascii=False)
        return citations_df

    def get_citing_insides_and_edges(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Process stored citations to filter only internal citations and construct edges.
        Returns a DataFrame of internal citation counts and an edges array.
        """
        # Load previously fetched citations
        cite_path = os.path.join(self.path, self.folder, "pd_articles_citations.json")
        article_citing = pd.read_json(cite_path)

        # Prepare output structures
        processed_df = pd.DataFrame(columns=[
            'Paper ID', 'Citing papers', 'Num citations', 'Year'
        ])
        all_ids = article_citing['Paper ID'].tolist()
        id_set = set(all_ids)
        edges = np.empty((0, 2), dtype=object)
        start_time = time.time()

        for idx, row in enumerate(article_citing.itertuples()):
            pid = row._1  # Paper ID
            year = row._4  # Year
            citing_list = np.asarray(row._2)  # Citing papers list
            internal = citing_list[np.isin(citing_list, all_ids)]

            processed_df = processed_df.append({
                'Paper ID': pid,
                'Citing papers': internal,
                'Num citations': len(internal),
                'Year': np.float32(year)
            }, ignore_index=True)

            # Build edge list
            if len(internal) > 0:
                src = np.full(len(internal), pid, dtype=object)
                edges = np.vstack((edges, np.vstack((src, internal)).T))

            # Periodic save
            if idx % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Index: {idx}/{len(all_ids)} time: {elapsed:.2f}s")
                processed_df.to_json(
                    os.path.join(self.path, self.folder, "articles_pd_citations_processed.json"),
                    orient='records', force_ascii=False
                )
                pd.DataFrame(edges).to_json(
                    os.path.join(self.path, self.folder, "edges.json"),
                    orient='records', force_ascii=False
                )
                start_time = time.time()

        # Final save
        processed_df.to_json(
            os.path.join(self.path, self.folder, "articles_pd_citations_processed.json"),
            orient='records', force_ascii=False
        )
        pd.DataFrame(edges).to_json(
            os.path.join(self.path, self.folder, "edges.json"),
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
        existing = set(references_df['Paper ID'].tolist())
        for idx, row in tqdm(articles_df.iterrows(), total=len(articles_df), desc='References'):
            pid = row['Paper ID']
            if pid in existing:
                continue
            # Fetch single work details
            self._enforce_rate_limit()
            url = f"{self.BASE_URL}/{pid}"
            resp = requests.get(url, params={'mailto': self.email})
            resp.raise_for_status()
            work = resp.json()
            refs = [r.split('/')[-1] for r in work.get('referenced_works', [])]
            references_df = references_df.append({
                'Paper ID': pid,
                'References': refs,
                'Num references': len(refs),
                'Year': row.get('Year')
            }, ignore_index=True)
            existing.add(pid)
            if len(references_df) % save_every == 0:
                references_df.to_json(ref_path, orient='records', force_ascii=False)
        # Final save
        references_df.to_json(ref_path, orient='records', force_ascii=False)
        return references_df

