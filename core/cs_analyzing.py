# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:37:16 2023

@author: jarom
"""


import json
import pandas as pd
import igraph as ig
import xnetwork as xn
import time
import pickle

import os
import cv2
from nltk.corpus import stopwords;
from nltk.stem.wordnet import WordNetLemmatizer;
import nltk.data;
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize;
from nltk.corpus import wordnet;
import re
import nltk
from infomap import Infomap

import torch
torch.cuda.is_available()
print(str(torch.version.cuda))

import numpy as np
import operator
from os.path import join as PJ

#kw_model = KeyBERT()

from keybert import KeyBERT
from keybert.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer

from scipy.interpolate import interpn
import requests
from pyalex import Works, Authors, Institutions, Concepts
import pyalex
import datetime
import traceback
import sys
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize 
from matplotlib import cm
from tqdm import tqdm
import random

class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model
        self.pool = self.embedding_model.start_multi_process_pool()

    def embed(self, documents, verbose=False):
        embeddings = self.embedding_model.encode_multi_process(documents,self.pool) 
        return embeddings
        
class Support():
    import os
    def __init__(self,path):
        self.path=path
    
    def read_files_to_list(self, folder):
        """ Input in string of path (folder)"""
        images = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
        self.path_file=images
        
    def save_dict(self,save_path,dictionary,name_of_dictionary):  
        pickling_on = open(self.path+save_path+name_of_dictionary+".txt","wb")
        pickle.dump(dictionary, pickling_on)
        pickling_on.close() 
         
    def load_dict(self,load_path,name_of_dictionary): 
        print(self.path+load_path+name_of_dictionary+".txt")
        pickle_off = open(self.path+load_path+name_of_dictionary+".txt", 'rb')
        dictionary = pickle.load(pickle_off)
        return dictionary    

class Processing_articles():
    """ 
    Need to implement algorithm from OpanAlex, where:
        1. Download articles with authors (OpenAlex_v2                          -> aritlces_pd, authors_pd)         :DONE                      ->Succes in: def make_DB_articles(self,email,searched_keywords,search_articles):
        2. Download all citations for articles (OpenAlex_v2)                    -> articles_pd_citations            :DONE                      ->Succes in: def get_articles_citations(self):
        3. Generate citations for articles included     
           articles_pd_citations - subcitations (OpenAlex_v2)                   -> articles_pd_citations_v1         :DONE                      ->Succes in: get_citing_insideandedges()
        4. According to subcitations make edges - articles_pd_citations_v1      -> edges                            :DONE                      ->Succes in: get_citing_insideandedges()
        5. Tokenized aricles by KeyBERT                                         -> articles_pd_keywords             :DONE                      ->Succes in: def tokenizing(self,path_articles,path_artickes_tokens):
        6. Generate the basic network                                           -> make_xnetwork                    :DONE                      ->Succes in: def make_xnetwork(self,forbidden_nodes,export_name_xnetwork, verbose_edgetitles=False):
        7. Clustering the network                                               -> make_xnetwork_clustered          :DONE                      ->Succes in: def make_xnetwork_clustered(self,network):
        8. Cosinus similarity                                                   -> cosinus similarity by sciBERT    :DONE Not yet implemented  ->Get to def()
        9. Worldmap - institutions, citations                                   -> by geopandas make worldmap       :DONE Not yet implemented  ->Get to def()
        10. NPMI or UMass                                                       -> Evaluation mwteircis to the text :NEED to implement         ->Need to develope
        11. UMAP                                                                -> Need to investigate              :NEED to investigate       ->?                                      
    """
    
    def __init__(self,path,folder,name_of_network):
        self.path=path
        self.folder=folder
        self.export_name_xnetwork=""
        self.kw_model = KeyBERT()
        try:
            self.articles_pd=pd.read_json(self.path+self.folder+"pd_articles.json")
            self.papers_ids=np.asarray(self.articles_pd["Paper ID"])
            self.papers_tis=np.asarray(self.articles_pd["Title"])
        except:
            msg=self.path+self.folder+"pd_articles.json"
            print(f"Do not exist: {msg}")
        try:
            self.authors_pd=pd.read_json(self.path+self.folder+"pd_authors.json")
        except:
            msg=self.path+self.folder+"pd_authors.json"
            print(f"Do not exist: {msg}")
        try:
            self.all_citations=pd.read_json(self.path+self.folder+"pd_articles_citations.json")
        except:
            msg=self.path+self.folder+"pd_articles_citations.json"
            print(f"Do not exist: {msg}")
        try:    
            self.all_citations_processed=pd.read_json(self.path+self.folder+"articles_pd_citations_processed.json") #change to:"D:/Projekty/2023_Brazilia/bardosova_project-main/articles_pd_citations_processed.json"
        except:
            msg=self.path+self.folder+"articles_pd_citations_processed.json"
            print(f"Do not exist: {msg}")
        try:
            self.all_keywords=pd.read_json(self.path+self.folder+"pd_articles_keywords.json")           
        except:
            msg=self.path+self.folder+"pd_articles_keywords.json"
            print(f"Do not exist: {msg}")            



    
    def init_ProgressBar(self,est_time=30,finish_value=30,):
        self.pb=ProgressBar(0, 1)
        self.pb.finish_value=finish_value
        self.pb.plot_progress()
        self.pb.loop_value=0
        
    def plot_ProgessBar(self,curProgress,actual_value):
        self.pb.actual_value=actual_value
        self.pb.set_and_plot(curProgress)


        
        
    def make_DB_articles(self,email,searched_keywords,search_articles):
        """
        email="jaromir.klarak@savba.sk" ------->string
        searched_keywords="wound*" ------->string
        search_articles=400  ----> integer
        
        """
        
        pyalex.config.email = email
        """
        1st - pd.concat() + list
        for i in range(10000):          4.88 s ± 47.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        
        2nd - pd.append() + dict
        for i in range(10000):          10.2 s ± 41.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        
        3rd - pd.DataFrame().loc + index operations
        for i in range(10000):          17.5 s ± 37.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        """

        """
        Making the database of articles for further processing. Due to huge data, 
        there is necessity store data in process of collection in way of save to .json files.
        
        There was changes in OpenAlex form of database. There are posibilities to changes in way of storing data
        """
        path=self.path
        folder=self.folder
    
        results = Works().get()
        results, meta = Concepts().get(return_meta=True)
        
        try:
            articles_pd = pd.read_json(path+folder+"pd_articles.json")
            print(f"Loaded articles_pd pandas Dataframe 'articles_pd' from: {path+folder}pd_articles.json")
        except:
            articles_pd=pd.DataFrame(columns=['Paper ID',"Title",'Abstract','Year','Date of publish','No. of citations','Journal','Journal ID', "ISSN", "Type", "Publisher", "Doi"])
            print("Created empty articles_pd pandas Dataframe 'articles_pd'")
        
        try:
            authors_pd = pd.read_json(path+folder+"pd_authors.json")
            print(f"Loaded authors_pd pandas Dataframe 'authors_pd' from: {path+folder}pd_authors.json")
            
        except:
            authors_pd=pd.DataFrame(columns=['Paper ID',"Title",'Author','Author ID','Author ORCID','Institution','Institution ID', "Country","Authors place",'Year','No. of citations','Journal', "ISSN", "Type", "Publisher"])
            print("Created empty authors pandas Dataframe 'authors_pd'")
        
        stored_db=len(articles_pd)
        stored_db_pre=len(articles_pd)
        page_len=0
        page_num=0
        all_papers=0
        indexed_papers=0
        all_authors=0
        indexed_authors=0
        main_index=0
        sleep_index=0
        start_timeg=time.time()
        for kw in searched_keywords:
            pager = Works().search_filter(abstract=(kw)).paginate(per_page=200, n_max=search_articles)
            start_time=time.time()
            for page in tqdm(pager):
                page_num+=1
                page_len+=len(page)
                if page_len>stored_db_pre:
                    for idx, paper_dict in enumerate(page):
                        main_index+=1
                        try:
                            paper_idx_id=paper_dict["id"].replace("https://openalex.org/", "")
                            time_year, year=self.time_str2date(paper_dict)
                            
                            paper_idx_id in articles_pd['Paper ID']
                            all_papers+=1

                            try:
                                articles_idx={'Paper ID':paper_idx_id,"Title":remove_non_ascii(str(paper_dict["title"])),'Abstract':Works()[paper_idx_id]["abstract"],'Year':time_year,'Date of publish':year, 
                                                           'No. of citations':paper_dict["cited_by_count"],'Journal':paper_dict["primary_location"]["source"]["display_name"] ,'Journal ID':str(paper_dict["primary_location"]["source"]["id"]).replace("https://openalex.org/", ""),
                                                           "ISSN":paper_dict["primary_location"]["source"]["issn_l"] , "Type":paper_dict["primary_location"]["source"]["type"], "Publisher":paper_dict["primary_location"]["source"]["host_organization_name"], "Doi":paper_dict["ids"]["doi"]}
                                articles_idx_pd=pd.DataFrame(articles_idx,index=[0])    
                                articles_pd=pd.concat([articles_pd, articles_idx_pd], ignore_index=True)
                                indexed_papers+=1
                               
                            except Exception:
                                #traceback.print_exc()
                                #print("Wrong in articles: "+str(paper_idx_id))
                                continue
                
                            ################################################################################################################################################################################################
                            for idxx,author_dict in enumerate(paper_dict["authorships"]):  
                                all_authors+=1
                                try:
                                    if "middle"==author_dict["author_position"]:
                                        author_place=str(idxx+1)
                                    else:
                                        author_place=author_dict["author_position"]
                                    authors_idx={'Paper ID':paper_idx_id,"Title":remove_non_ascii(str(paper_dict["title"])),'Author':author_dict["author"]["display_name"],'Author ID':author_dict["author"]["id"].replace("https://openalex.org/", ""),
                                                 'Author ORCID':author_dict["author"]["orcid"],'Institution':author_dict["institutions"][0]["display_name"],'Institution ID':author_dict["institutions"][0]["id"].replace("https://openalex.org/", ""), 
                                                 "Country":author_dict["institutions"][0]["country_code"],"Authors place":author_place,'Year':[year], 
                                                 'No. of citations':paper_dict["cited_by_count"],'Journal':paper_dict["primary_location"]["source"]["display_name"],
                                                 "ISSN":paper_dict["primary_location"]["source"]["issn_l"] , "Type":paper_dict["primary_location"]["source"]["type"], "Publisher":paper_dict["primary_location"]["source"]["host_organization_name"]}
                                      
                                except Exception:
                                    #traceback.print_exc()
                                    #print("Wrong authors: "+str(paper_idx_id))
                                    if type(author_dict["author"]["display_name"])==str and len(author_dict["author"]["display_name"])>0:
                                        authors_idx={'Paper ID':paper_idx_id,"Title":[str(paper_dict["title"])],'Author':author_dict["author"]["display_name"]}
                                         
                                    
                                    if type(author_dict["author"]["id"])==str and len(author_dict["author"]["id"])>20:
                                            authors_idx={'Author ID':author_dict["author"]["id"].replace("https://openalex.org/", "")}
                                              
                                    else:
                                        authors_idx={'Paper ID':paper_idx_id,"Title":[str(paper_dict["title"])],'Author':"Empty Name"}
                                         
                                
                                    continue
                                authors_idx_pd=pd.DataFrame(authors_idx)
                                authors_pd=pd.concat([authors_pd, authors_idx_pd], ignore_index=True)
                                indexed_authors+=1
                
                        except:
                            #print("Error with something in: "+str(paper_idx_id)+" in "+str(index_art))
                            continue
            
                    if main_index%500==0:
                        stored_db=len(articles_pd['Paper ID'].values)
                        print(f"Already stored: {stored_db}  Progress: {main_index}/{search_articles} Indexed articles: {indexed_papers}/{all_papers} Indexed authors: {indexed_authors}/{all_authors} in time: {time.time()-start_time}") 
                        """ Export pandas DataFrame to json file"""
                        articles_pd.to_json(r""+path+folder+"pd_articles.json")
                        authors_pd.to_json(r""+path+folder+"pd_authors.json")
                        start_time=time.time()
                        all_papers=0
                        indexed_papers=0
                        all_authors=0
                        indexed_authors=0
                    if sleep_index%95000:
                        actual_time=time.time()-start_timeg
                        print(f"Overcross number  of requests... \nTime to sleeping: \t{np.round(24*60*60-actual_time,2)} \nIn time: \t\t\t{datetime.datetime.fromtimestamp(time.time())}")
                        time.sleep(24*60*60-actual_time)
                        start_timeg=time.time()
                        sleep_index=0

        articles_pd.to_json(r""+path+folder+"pd_articles.json")
        authors_pd.to_json(r""+path+folder+"pd_authors.json")

        
        
    def get_citing_papers(self, paper_str):
        
        # url with a placeholder for cursor
        """ paper_dict is get from pagination (function 'paginate')"""
        if type(paper_str) != str:
            raise AttributeError("Variable 'paper_str' is not type 'string'! ")
            
        #for dictionary
        #paper_idx_cited_url=paper_dict["cited_by_api_url"]+str("&per-page=200&cursor={}")
        
        # for string
        paper_idx_cited_url="https://api.openalex.org/works?filter=cites:"+paper_str+str("&per-page=200&cursor={}")
        #per-page=200
        cursor = '*'
    
        cited_paper=np.asarray([])
        # loop through pages
        idx=0
        while cursor:
            # set cursor value and request page from OpenAlex
            url = paper_idx_cited_url.format(cursor) # new
            #print("\n" + url)
            page_with_results = requests.get(url).json()
    
            # loop through partial list of results
            results = page_with_results['results']
            if len(results)>0:
                results_pd=pd.DataFrame(results)["id"]
                results_pd_test=str(list(results_pd)).replace('https://openalex.org/', '')
                
                results_pd_filt = re.sub("['']", "", results_pd_test)
                results_pd_filt=results_pd_filt.replace("[","")
                results_pd_filt=results_pd_filt.replace("]","")
                results_pd_list=np.asarray(results_pd_filt.split(", "))
                
          
                cursor = page_with_results['meta']['next_cursor']
            else:
                print("Break")
                results_pd_list=np.asarray([])
                break
            if idx==0:
                cited_paper=np.copy(results_pd_list)
            else:
                cited_paper=np.concatenate([cited_paper,results_pd_list],axis=0)
            idx+=1
                
        return cited_paper

    def get_articles_citations(self):
        proart_path=self.path
        proart_folder=self.folder
        
        articles_pd = self.articles_pd
        try:
            self.article_citing=pd.read_json(proart_path+proart_folder+"pd_articles_citations.json")
            print("Read -articles_pd_citations- from json file")
        except:    
            self.article_citing=pd.DataFrame(columns=["Paper ID", "citing papers","Num citations", "Year"])
            print("Created emtpy variable article_citing  for articles_pd_citations.json")
        
        start_time=time.time()
        len_loop=len(articles_pd["Paper ID"])
        cumulative_time=1/len_loop
        self.init_ProgressBar(30,len_loop)
        self.pb.loop_value=100
        
        for jdxx,paper_idx_id in enumerate(list(articles_pd["Paper ID"])):
            try:
                bool_loop=(self.article_citing['Paper ID'].str.contains(paper_idx_id).any())
            except:
                bool_loop=False
            if (not bool_loop==True):
                try:
                    citing_papers=np.asarray(self.get_citing_papers(paper_idx_id))
                    self.article_citing.loc[len(self.article_citing),["Paper ID", "citing papers","Num citations","Year"]]=[paper_idx_id,citing_papers,len(citing_papers),(articles_pd.loc[jdxx]["Year"])]
                    #print("Index: %s, Number of citation: %s--- %s seconds ---" % (jdxx, len(citing_papers),(time.time() - start_time)))
                except:
                    pass
            #else:
            #   print(f"Article is yet indexed in database: {paper_idx_id}")
        
            """ Progressbar"""
            loop_time=(time.time() - start_time)
            cumulative_time+=loop_time
            estimate_time=str(datetime.timedelta(seconds=int((len_loop-jdxx)*(cumulative_time/(jdxx+1)))))
            current_progress=int(30*(jdxx/len_loop))+1
            
            self.pb.print_message=str("Est. Time: "+str(estimate_time)+"   Time (100 loops): "+str(int(cumulative_time*100/(jdxx+1)))+" s     ")
            self.plot_ProgessBar(current_progress,jdxx)
            start_time=time.time()
            """ End of ProgressBar"""
            if jdxx%1000==0:
                """ Export pandas DataFrame to json file"""
                try:
                    self.article_citing.to_json(r""+self.path+self.folder+"pd_articles_citations.json")
                except:
                    print(f"Do not save! {jdxx}/{len_loop}")
                    
                    
    def get_citing_insideandedges(self):
        article_citing=pd.read_json(self.path+self.folder+"pd_articles_citations.json")
        article_citing_v2=pd.DataFrame(columns=["Paper ID", "citing papers","Num citations", "Year"])
        citing_list=list(article_citing["Paper ID"])
        citing_numpy=np.asarray(citing_list)
        start_time = time.time()
        list_papers=list(article_citing["Paper ID"])
        edges_np=np.asarray([[],[]],dtype="object").T
        for pdx,article_pdx in enumerate(list_papers):
            citing_papers_pdx=np.asarray(list((article_citing.iloc[pdx]["citing papers"])))
            citations=citing_papers_pdx[np.isin(citing_papers_pdx,citing_numpy)]
            article_citing_v2.loc[len(article_citing_v2),["Paper ID", "citing papers","Num citations", "Year"]]=[article_pdx,citations,len(citations),np.float32(article_citing.iloc[pdx]["Year"])]
            #article_citing_v2.loc[len(article_citing_v2.index)]=[article_pdx,citations,len(citations),(article_citing.iloc[pdx]["Year"])]

            edges_pdx2=np.zeros((len(citations),2),dtype="object")
            edges_pdx2[:,0]=article_pdx
            edges_pdx2[:,1]=citations
            #edges_pdx=np.concatenate((ed1,citations),axis=1)
            edges_np =np.concatenate((edges_np,edges_pdx2),axis=0)
            
            edges=pd.DataFrame({"0":edges_np[:,0],"1":edges_np[:,1]})
            if pdx%100==0:
                print(f"index: {pdx} / {len(article_citing)} \t time:  {(time.time() - start_time)} sec.") 
                article_citing_v2.to_json(r""+self.path+self.folder+"articles_pd_citations_processed.json") 
                edges.to_json(r""+self.path+self.folder+"edges.json") 
                start_time = time.time()
        article_citing_v2.to_json(r""+self.path+self.folder+"articles_pd_citations_processed.json") 
        edges.to_json(r""+self.path+self.folder+"edges.json")   


    def tokenizing(self):
        """
        For tokenized all articles, make json file with tokens 
        input are in:
            path_articles           - location of json files with articles, mainly (Paper ID, Title and abstract)
            path_artickes_tokens    - location of json files with tokens for articles, mainly (Paper ID, Title and abstract and tokens and distances)
        """  
        articles_pd=self.articles_pd
        articles_heads = list(articles_pd.columns)   
            
        try:
            tokenized_articles=pd.read_json(self.path+"pd_articles_keywords.json")
            print("Loaded keywords from json saved file to tokenizing abstracts by -Processing_articles.tokenizing-")
        except:
            tokenized_articles=pd.DataFrame(columns=["tokens_id","tokens_titles","tokens_abstracts","tokens_article","tokens_distance"])
            print("Initialized empty datagrame for keywords to tokenizing abstracts by -Processing_articles.tokenizing-")
            

        tokens_names=list(tokenized_articles.columns)
        start_time=time.time()
        list_tokenized=list(tokenized_articles[tokens_names[0]])
        
        for article_idx in range(len(articles_pd)):

            article_id=str(articles_pd.iloc[article_idx][articles_heads[0]])

            
            if article_id not in list_tokenized:
                article_title=str(articles_pd.iloc[article_idx][articles_heads[1]])
                article_abstract=str(articles_pd.iloc[article_idx][articles_heads[2]])
                
                tokennized_strings,tokennized_strings_di=self.tokenizeString_v2(article_abstract,30,article_title)
                
                tokenized_articles.loc[article_idx,tokens_names]=[article_id,article_title,article_abstract,tokennized_strings,tokennized_strings_di]
                
            if(article_idx%100==0):
                print(f"{article_idx}/{len(articles_pd)} \t Time: {np.round((time.time() - start_time),4)} sec.")
                start_time=time.time()
                tokenized_articles.to_json(self.path+self.folder+"pd_articles_keywords.json")
        tokenized_articles.to_json(self.path+self.folder+"pd_articles_keywords.json")
        return tokenized_articles
    
    def make_xnetwork(self,forbidden_nodes,export_name_xnetwork,edges_name, verbose_edgetitles=False):
        """
        forbidden_nodes = 1D numpy array of forbidden articles in way of "Paper ID" 
        verbose_edgetitles = making of list of citing article-cited articles based on edges co back check correctly method
        export_name_xnetwork = string, name of export xnetwork
        edges_name = string, 2 possibilities: edges1 or edges2
        """
        
        """
        "from __init__(self):"
        articles_pd                 = proart.articles_pd=pd.read_json("D:/Projekty/2023_Brazilia/bardosova_project-main/saved/articles_pd.json")
        authors_pd                  = proart.authors_pd=pd.read_json("D:/Projekty/2023_Brazilia/bardosova_project-main/saved/authors_pd.json")
        all_citations               = proart.all_citations=pd.read_json("D:/Projekty/2023_Brazilia/bardosova_project-main/saved/articles_pd_citations.json")
        all_citations_processed     = proart.all_citations_processed=pd.read_json("D:/Projekty/2023_Brazilia/bardosova_project-main/saved/articles_pd_citations_wound.json")
        """
        self.export_name_xnetwork=export_name_xnetwork
        print("Processing_articles----> def make_xnetwork()")
        
        """ Initializing data """
        edges=pd.read_json(self.path+self.folder+edges_name+".json")
        """ End of initializing """
        
        
        columns=edges.columns
        edges_np=np.asarray(edges[columns])
        
        x=np.asarray(forbidden_nodes)
        yy=edges_np
        #x=xx
        index = np.argsort(x)
        sorted_x = x[index]
        #for i in range(2):
            
        y=edges_np[:,0]
        y=yy[:,0]
        sorted_index = np.searchsorted(sorted_x, y)
        yindex = np.take(index, sorted_index, mode="clip")
        mask1 = x[yindex] != y
        posi1=np.where(mask1!=False)[0]
        
        y=edges_np[:,1]
        y=yy[:,1]
        sorted_index = np.searchsorted(sorted_x, y)
        yindex = np.take(index, sorted_index, mode="clip")
        mask2 = x[yindex] != y
        posi2=np.where(mask2!=False)[0]
        

        posi=np.intersect1d(posi1, posi2)

        ynew=yy[posi,:]
        list_new_edges2=ynew
        list_new_edges1=np.vstack([list_new_edges2[:,0],list_new_edges2[:,1]]).T
        self.list_new_nodes1=np.unique(list_new_edges1,return_counts="True")[0]

        """ Find Titles of articles for nodes to network"""
        self.finded_titles=self.match_numpy(self.list_new_nodes1, np.asarray(self.papers_ids), np.asarray(self.papers_tis), bool_value=True)


        if verbose_edgetitles==True:    
            self.list_new_edges3=list()
            for idx,item in enumerate(list_new_edges2):
                posi1=np.where(item[0]==self.papers_ids)[0]
                posi2=np.where(item[1]==self.papers_ids)[0]
                
                self.list_new_edges3.append([remove_non_ascii(str(self.papers_tis[posi1])), remove_non_ascii(str(self.papers_tis[posi2]))])
               
                
        print(f"Number of edges: {list_new_edges1.shape[0]}")
        g = ig.Graph(directed=True)
        g.add_vertices(len(self.list_new_nodes1))
        g.vs['name'] = list(self.list_new_nodes1)
        g.vs['title'] = list(self.finded_titles)
        g.add_edges(list_new_edges1)

        print(g.vcount())
        print(g.ecount())
        """http://localhost:5173/docs/example/index.html?network=toy_citation_network"""
        print("Xnetwork")
        xn.igraph2xnet(g, self.path+"network_veri2.xnet")

        network = xn.xnet2igraph(self.path+"network_veri2.xnet")
        network.vs['wos_id'] = network.vs['name']
        network.vs['name'] = network.vs['title']

        xn.igraph2xnet(network,self.path+str(export_name_xnetwork)+".xnet",ignoredNodeAtts=["Text"])
        if verbose_edgetitles:
            return None
        else:
            return self.list_new_nodes1

    def match_numpy(self,y,x,arr_exp,bool_value):
        """
        Input values in way:
            y - sub array (find the values)
            x - primary array (where are finding values)
            arr_exp - explored array, chooese values according to array x and y
            bool_value - The way, where are exported values primary finding or supress values from sub array (x) 
            
        For reverse, change inputs for "x" and "y" variables and set boll functions as "True"
        """
        print("\t  Processing_articles ----> def match_numpy()")
        """
        y=papers_ids
        x=xport
        """
        
        index = np.argsort(x)
        sorted_x = x[index]
        sorted_index = np.searchsorted(sorted_x, y)
        yindex = np.take(index, sorted_index, mode="clip")
        mask = x[yindex] != y
        mask = x[yindex] != y
        self.posi = np.ma.array(yindex, mask=mask)
        
        self.posi_inverse=np.where(mask!=bool_value)[0]
        
        self.ynew=arr_exp[self.posi] 
        self.ynew_reverse=arr_exp[self.posi_inverse] 
        
        return self.ynew
        
    def get_uni_values(self,pandas_keywords):
        """
        pandas_keywords = pd.read_json("D:/Projekty/2023_Brazilia/bardosova_project-main/saved/articles_pd_keywords.json")  
                     = Must be pandas dataframe of keywords and contain column - ["tokens_article"]!
                 
        the paralelization of get key values from very large dataset. There was effort built on numpy library, but withou succes. There was probelm build large numpy array od UXXX (U100 - unicode string)
        high consumption of memory. In case of dtpe object for numpy was suitable but without possibility to use function numpy.unique()
        Instead of numpy was used pandas as suitable posibility to paralized keywords in very large dataset
        """
        
        #pandas_keywords=pd.read_json("D:/Projekty/2023_Brazilia/bardosova_project-main/saved/articles_pd_keywords.json")
        print("Processing_articles ----> def get_uni_values()")
        
        whole_tokens_list=list(pandas_keywords["tokens_article"])
        """Proccesing the kyewords to numpy array"""
        new_numpy=list()
        start_time=time.time()
        new_numpy=np.zeros((len(whole_tokens_list),90),dtype="object")

        for idx, item in enumerate(whole_tokens_list):
            item_np=np.asarray(item)
            new_numpy[idx,:len(item_np)]=item_np

        new_numpy_pd=pd.DataFrame(new_numpy)
        columns=new_numpy_pd.columns
        print(f"\nProcessing_articles ----> def get_uni_values()  Initiliazed large numpy array     Time: {time.time() - start_time} sec.\n")
        
        start_time=time.time()
        for idx,item in enumerate(columns):
            if idx==0:
                pd_new=new_numpy_pd[item].copy()
            else:
                pd_new=pd.concat([pd_new, new_numpy_pd[item]])
            if idx%10==0 and len(whole_tokens_list)>200000:
                print(f"Processing_articles ----> def get_uni_values()  Index: {idx}/{len(columns)} Time: {time.time() - start_time} sec.")
                start_time=time.time()
            elif idx==len(columns):
                print(f"Processing_articles ----> def get_uni_values()  Successful finished! Time: {time.time() - start_time} sec.")
                start_time=time.time()                
                
        if len(pandas_keywords)==len(self.papers_ids):
            self.keywords_uni_whole=pd_new.value_counts()
            self.keywords_uni=self.keywords_uni_whole.copy()
        else:
            self.keywords_uni=pd_new.value_counts()
        return self.keywords_uni
        
    def extract_keywords(self,network):
        #all_keywords=self.all_keywords
        #edgelist = [(e.source,e.target) for e in network.es]
        communitiespre = self.communitiespre
        communities_np=np.asarray(communitiespre).astype(np.int32)
        self.communities_np_uni=np.unique(communities_np,return_counts="True")
        nodes=np.asarray(network.vs['wos_id'])

        x=np.asarray(self.papers_ids)
        arr_exp=np.arange(len(x))
        self.keywords_uni_corpus=self.get_uni_values(self.all_keywords)
        self.keywords_clusters={}
        for idx, item in enumerate(self.communities_np_uni[0]):
            posi_articles=np.where(communities_np==item)[0]
            nodes_idx=nodes[posi_articles]
            
            posi_keywords_idx=self.match_numpy(nodes_idx, x, arr_exp, None)
            print(f"Processing_articles ----> def extract_keywords(): {idx}/{len(self.communities_np_uni[0])}")
            
            keywords_idx=self.all_keywords.loc[posi_keywords_idx]
            keywords_uni_idx=self.get_uni_values(keywords_idx)
            self.keywords_clusters[idx]=keywords_uni_idx
            
        return self.keywords_clusters

    def make_xnetwork_clustered(self,network):
        print("Processing_articles ----> def make_xnetwork_clustered()")
        start_time_network=time.time()
        self.defaultNames = "ABCDEFGHIJKLMNOPQRSTUWVXYZ"
        
        self.list_new_nodes1=network.vs['wos_id']
        nodes=self.list_new_nodes1
        edgelist = [(e.source,e.target) for e in network.es]
        communitiespre = infomapApply(network)[0]
        self.communitiespre=communitiespre
        communities_np=np.asarray(communitiespre).astype(np.int32)
        communities_np_uni=np.unique(communities_np,return_counts="True")
        self.communities_np=communities_np
        nodes=np.asarray(network.vs['wos_id'])
        nodes_name=np.asarray(network.vs['name'])
        
        
        keywords_clusters=self.extract_keywords(network)
        keywords_uni_corpus=self.keywords_uni_corpus  
        
        clusters_np=list()
        
        start_time=time.time()
        len_loop=len(self.communities_np_uni[0])
        cumulative_time=1/len_loop
        self.init_ProgressBar(30,len_loop)
        self.pb.loop_value=100
        for idx, uni_idx in enumerate(self.communities_np_uni[0]):
            clusters_positions=np.where(uni_idx==communities_np)[0]
            clusters_np.append(clusters_positions)
            
            " Progressbar"
            loop_time=(time.time() - start_time)
            cumulative_time+=loop_time
            estimate_time=str(datetime.timedelta(seconds=int((len_loop-idx)*(cumulative_time/(idx+1)))))
            current_progress=int(30*(idx/len_loop))+1
            
            self.pb.print_message=str("Est. Time: "+str(estimate_time)+"   Time (100 loops): "+str(int(cumulative_time*100/(idx+1)))+" s ---> Get clusters    ")
            self.plot_ProgessBar(current_progress,idx)
            start_time=time.time()
        
        start_time = time.time()
        verticesCount=network.vcount()
        tokenImportanceIndexInClusters = []
        reports_tokens = {}
        
        start_time=time.time()
        len_loop=len(self.communities_np_uni[0])
        cumulative_time=1/len_loop
        self.init_ProgressBar(30,len_loop)
        self.pb.loop_value=100
        
        for clusterIndex in range(len(self.communities_np_uni[0])):
            
            tokenImportanceIndexInCluster = pd.DataFrame() #columns=["count"]
            token_computing_importance_idx=pd.DataFrame()
            
            clusterSize = len(clusters_np[clusterIndex])
            tokenFrequencyInCluster = keywords_clusters[(clusterIndex)]
            
            cluster_new_tokens=np.asarray(keywords_clusters[(clusterIndex)].index)
            tokenFrequencyInCluster=pd.DataFrame(tokenFrequencyInCluster)
            columns=list(pd.DataFrame(tokenFrequencyInCluster).columns)
            
            
            nInCluster = np.asarray(tokenFrequencyInCluster[columns])
            nOutCluster = np.asarray(keywords_uni_corpus[cluster_new_tokens]).flatten()-nInCluster.flatten()
            outClusterSize = verticesCount-clusterSize
        
            FInCluster = (nInCluster)/(clusterSize)
            FOutCluster = (nOutCluster)/(outClusterSize)
            
            FInCluster1 = (nInCluster)/(np.sum(nInCluster))
            FOutCluster1 = (nOutCluster)/(np.sum(nOutCluster))
            
            importanceIndex = FInCluster.flatten()-FOutCluster.flatten()
            importanceIndex1 = (1000*(FInCluster.flatten()-FOutCluster.flatten()))*(nInCluster.flatten()**2/np.sum(nInCluster))
            
            
            cluster_size_list=np.zeros((len(nInCluster)),dtype="int32")
            cluster_size_list[:]=clusterSize
            cluster_out_size_list=np.zeros((len(nInCluster)),dtype="int32")
            cluster_out_size_list[:]=outClusterSize
            
            
            token_computing_importance_idx["Indexes"]=list(tokenFrequencyInCluster.index)
            token_computing_importance_idx["Tokens in Cluster"]=list(tokenFrequencyInCluster["count"])
            token_computing_importance_idx["Size of Cluster"]=list(cluster_size_list)
            
            token_computing_importance_idx["Tok. in Out clusters"]=list(nOutCluster)
            token_computing_importance_idx["Size of Out Clusters"]=list(cluster_out_size_list)
            token_computing_importance_idx["Tokens in Whole"]=list(np.asarray(keywords_uni_corpus[cluster_new_tokens]).flatten())
            
            token_computing_importance_idx["FInCluster"]=list(FInCluster.flatten())
            token_computing_importance_idx["FOutCluster"]=list(FOutCluster.flatten())
            token_computing_importance_idx["FIn - FOut as Importance"]=list(importanceIndex)
            token_computing_importance_idx["New Importance"]=list(importanceIndex1)
        
            tokenImportanceIndexInCluster = pd.DataFrame() #columns=["count"]
            tokenImportanceIndexInCluster.index=list(cluster_new_tokens)
            #tokenImportanceIndexInCluster.loc[cluster_new_tokens] = importanceIndex.T
            tokenImportanceIndexInCluster.insert(0, "count", list(importanceIndex), True)
        
        
                
            #print(f"Index of clustering: {clusterIndex}/{len(clusters_np)} \t time:  {(time.time() - start_time)} sec.")
            start_time = time.time()
            reports_tokens[clusterIndex]=token_computing_importance_idx
            tokenImportanceIndexInClusters.append(tokenImportanceIndexInCluster)
            
            " Progressbar"
            loop_time=(time.time() - start_time)
            cumulative_time+=loop_time
            estimate_time=str(datetime.timedelta(seconds=int((len_loop-clusterIndex)*(cumulative_time/(clusterIndex+1)))))
            current_progress=int(30*(clusterIndex/len_loop))+1
            
            self.pb.print_message=str("Est. Time: "+str(estimate_time)+"   Time (100 loops): "+str(int(cumulative_time*100/(clusterIndex+1)))+" s ---> Get clusters    ")
            self.plot_ProgessBar(current_progress,clusterIndex)
            start_time=time.time()
            
            """   ENd of version wirh importcance clusters """
        
        self.reports_tokens=reports_tokens
        self.report_dict2xlsx(reports_tokens,self.export_name_xnetwork,100)
        
        
        minKeywordsPerCluster = 20;
        maxKeywordsPerCluster = 20;
        maxClusterNameLength = 150;
        
        
        defaultNamesLength = len(self.defaultNames);
        
        clusterKeywords = [];
        minClusterSize = min([len(cluster) for cluster in clusters_np]);
        maxClusterSize = max([len(cluster) for cluster in clusters_np]);
        clusterNames = [];
        start_time = time.time()
        loop_time=time.time()
        num_loops=0
        
        start_time=time.time()
        len_loop=len(self.communities_np_uni[0])
        cumulative_time=1/len_loop
        self.init_ProgressBar(30,len_loop)
        self.pb.loop_value=100
        
        for clusterIndex in range(len(self.communities_np_uni[0])):
            cluster = clusters_np[clusterIndex];
            clusterSize = len(cluster);
            keywords=list(tokenImportanceIndexInClusters[clusterIndex].sort_values(by=['count'], ascending=False).index)
        
            if(maxClusterSize>minClusterSize):
                m = (maxKeywordsPerCluster-minKeywordsPerCluster)/float(maxClusterSize-minClusterSize);
            else:
                m=0;
            keywordsCount = round(m*(clusterSize-minClusterSize)+minKeywordsPerCluster);
            currentKeywords = [];
            num_loops+=1
            if (loop_time-time.time())>5:
                print(f"Index of clustering: {clusterIndex}/{len(clusters_np)} \t Average time:  {np.round((loop_time - start_time)/num_loops,3)} sec. Loops: {num_loops}")
                num_loops=0
                loop_time=time.time()
            start_time = time.time()
            while(len(currentKeywords)<keywordsCount and len(keywords)>len(currentKeywords)):
                currentKeywords = keywords[0:keywordsCount]
                currentKeywords=list(remove_non_ascii(str(v)) for v in currentKeywords)
                jointKeywords = "."+".".join(currentKeywords)+".";
                toRemoveKeywords = [];
                try:
                    for keyword in currentKeywords:
                        if(jointKeywords.find(" %s."%keyword)>=0):
                            toRemoveKeywords.append(keyword);
                        elif(jointKeywords.find(".%s "%keyword)>=0):
                            toRemoveKeywords.append(keyword);
                    for toRemoveKeyword in toRemoveKeywords:
                        keywords.remove(toRemoveKeyword);
                        currentKeywords.remove(toRemoveKeyword);
                except:
                    pass
            clusterKeywords.append(currentKeywords);
            #print(currentKeywords);
            clusterName = "";
            if(clusterIndex<defaultNamesLength):
                clusterName += self.defaultNames[clusterIndex];
            else:
                clusterName += "{%d}"%(clusterIndex);
          
            clusterName += " - "+", ".join(currentKeywords);
            if(len(clusterName)>maxClusterNameLength):
                clusterName = clusterName[0:maxClusterNameLength-1]+"...";
            for vertexIndex in cluster:
                network.vs[vertexIndex]["Cluster Name"] = clusterName;
                network.vs[vertexIndex]["Cluster Index"] = clusterIndex;
            clusterNames.append(clusterName);
            print(clusterName);
            
            " Progressbar"
            loop_time=(time.time() - start_time)
            cumulative_time+=loop_time
            estimate_time=str(datetime.timedelta(seconds=int((len_loop-clusterIndex)*(cumulative_time/(clusterIndex+1)))))
            current_progress=int(30*(clusterIndex/len_loop))+1
            
            self.pb.print_message=str("Est. Time: "+str(estimate_time)+"   Time (100 loops): "+str(int(cumulative_time*100/(clusterIndex+1)))+" s ---> Get clusters    ")
            self.plot_ProgessBar(current_progress,clusterIndex)
            start_time=time.time()
            
            
        self.network_clusterd=network
        self.network_saved_name=self.path+self.folder+str(self.export_name_xnetwork)+"_clustered1A.xnet"
        xn.igraph2xnet(network,fileName=PJ(self.path+self.folder+str(self.export_name_xnetwork)+"_clustered1A.xnet"),ignoredNodeAtts=["Text"])
        print(f"Processing_articles ----> def make_xnetwork_clustered() ----> DONE in time: {np.round(time.time()-start_time_network,3)} sec.")

    def report_dict2xlsx(self, dictionary, name,border):
        print("Processing_articles ----> def report_dict2xlsx()")
        if type(dictionary)==dict and type(name)==str:
            names_columns=[]
            for udx,item in enumerate(dictionary[list(dictionary.keys())[0]]):
               names_columns.append(item)
               


               
            with pd.ExcelWriter(self.path+self.folder+name+".xlsx",engine="xlsxwriter") as writer:
               if border>len(list(dictionary.keys())):
                   border=len(list(dictionary.keys()))
                   
               if border>len(self.defaultNames):
                   cluster_indx_str=list(self.defaultNames)+list(np.arange(len(self.defaultNames),border))
               else:
                   cluster_indx_str=list(self.defaultNames)
               for idx in range(border):
                   pd_cluster_reports=dictionary[list(dictionary.keys())[idx]]
                   pd_cluster_reports1=pd_cluster_reports.sort_values(by=[names_columns[9]],ascending=False)[:200]

                   pd_cluster_reports1.to_excel(writer, sheet_name="cluster_"+str(cluster_indx_str[idx]), index=False)
            print("Processing_articles ----> def report_dict2xlsx() ------> DONE")
        else:
           print("Variable dictionary is not dict()!")
    
    def time_str2date(self, paper_dict):
        time_str=paper_dict["publication_date"]
        time_year=paper_dict["publication_year"]
        
        time_date=datetime.datetime.strptime(time_str, '%Y-%m-%d').date()
        
        if type(time_date)!=datetime.date:
            raise AttributeError(f"Variable 'time_date' is not type 'datetime.date'! \nit is type: '{type(time_date)}'")
        if type(time_year)!=int:
            raise AttributeError(f"Variable 'time_year' is not type 'int'! \nIt is type: '{type(time_year)}'")

        return time_year, time_date                   
    
    def tokenizeString_v2(self,article_abstract,maximumTokenSize,article_title):

        kb_keywords1SW=self.kw_model.extract_keywords(article_abstract, keyphrase_ngram_range=(1, 1),top_n=30, stop_words="english")
        kb_keywords2SW=self.kw_model.extract_keywords(article_abstract, keyphrase_ngram_range=(2, 2),top_n=30, stop_words="english")
        kb_keywords3SW=self.kw_model.extract_keywords(article_abstract, keyphrase_ngram_range=(3, 3),top_n=30, stop_words="english")
        
        tokens_wd = list()
        tokens_di = list()
        
        for wordIndex in range(3):
            item_list0=list()
            item_list1=list()
            for item in vars()["kb_keywords"+str(wordIndex+1)+"SW"]:
                if len(str(item[0]))>0 and type(np.float32(item[1]))!="np.float32":
                    item_list0.append(str(item[0]))
                    item_list1.append(np.float32(item[1]))
                else:
                    print(f"Wrong: {article_title} \t {item[0]} and {item[1]}")
            if len(item_list0)!=len(item_list1):
                print(f"Wrong: {article_title} \t {len(item_list0)} and {len(item_list1)}")

            tokens_wd+=item_list0
            tokens_di+=item_list1

        return tokens_wd,tokens_di

class ProgressBar(object):
    """Main class for the ProgressBa."""
    """
    init:
        start_time=time.time()
        len_loop=len(articles_pd["Paper ID"])
        cumulative_time=1/len_loop
        self.init_ProgressBar(30,len_loop)
        self.pb.loop_value=100
    
    loop:
        " Progressbar"
        loop_time=(time.time() - start_time)
        cumulative_time+=loop_time
        estimate_time=str(datetime.timedelta(seconds=int((len_loop-jdxx)*(cumulative_time/(jdxx+1)))))
        current_progress=int(30*(jdxx/len_loop))+1
        
        self.pb.print_message=str("Est. Time: "+str(estimate_time)+"   Time (100 loops): "+str(int(cumulative_time*100/(jdxx+1)))+" s     ")
        self.plot_ProgessBar(current_progress,jdxx)
        start_time=time.time()
    
    """
    DEFAULT_BAR_LENGTH = float(30)

    def __init__(self, start=0, step=1):
        """Init for the class."""
        self.end = ProgressBar.DEFAULT_BAR_LENGTH
        self.start = start
        self.step = step
        self.actual_value=0
        self.finish_value=0
        self.loop_value=0
        self.loop_time=0
        self.print_message=""
        self.saved_time=int(time.time())
        self.time_basic=int(time.time())
        self.total = self.end - self.start
        self.counts = self.total / self.step
        self._barLength = ProgressBar.DEFAULT_BAR_LENGTH

        self.set_level(self.start)
        self._plotted = False

    def set_level_old(self, level, initial=False):
        """Setting Level."""
        self._level = level
        if level < self.start:
            self._level = self.start
        if level > self.end:
            self._level = self.end

        self._ratio = float(
            self._level - self.start) / float(self.end - self.start)
        self._levelChars = int(self._ratio * self._barLength)

    def set_level(self, level, initial=False):
        """Setting Level."""
        self._level = level
        if level < self.start:
            self._level = self.start
        if level > self.end:
            self._level = self.end

        self._ratio = float(self._level) / float(self._barLength)
        self._levelChars = int(self._ratio * self._barLength) * self.step

    def plot_progress(self):
        """Plotting the bar."""
        sys.stdout.write("\r  %3i%%  |%s%s| %s %s " % (int(self._ratio * self.step * 100.0) , u'\u2588'*int(self._levelChars),' '*int(self._barLength - self._levelChars) ,
                                                   str(self.actual_value)+"/"+str(self.finish_value),self.print_message,))
        sys.stdout.flush()
        """
        var1=u'\u2588' * int(self._levelChars)
        var2=' ' * int(self._barLength - self._levelChars)
        sys.stdout.write(f"\r  {int(self._ratio * self.step * 100.0)}% |{var1}{var2}|")
        sys.stdout.flush()
        """
        self._plotted = True

    def set_and_plot(self, level):
        """Call the plot."""

        self.set_level(level)
        if (not self._plotted) or (self.saved_time != self.time_basic):
            self.plot_progress()
            self.saved_time=self.time_basic
        self.time_basic=int(time.time())

    def __del__(self):
        """Del for the class."""
        sys.stdout.write("\n")
               
def remove_non_ascii(string):
    new_string=string.encode('ascii', errors='ignore').decode()
    if type(new_string)!=str:
        new_string="0"
    return new_string

def infomapApply(g, weights=None):
    #g=network
    vertexCount = g.vcount()
    if(weights!=None):
        edges = [(e.source, e.target, e[weights]) for e in g.es]
    else:
        edges = g.get_edgelist()



    extraOptions = ""
    num_randoms=g.vcount()*g.ecount() #number of nodes and edges
    im = Infomap("%s -N 10 --silent --seed %d" %(extraOptions, np.random.uniform(g.vcount()*g.ecount())))
    
    im.setVerbosity(0)
    for nodeIndex in range(0, vertexCount):
        im.add_node(nodeIndex)
    for edge in edges:
        if(len(edge) > 2):
            if(edge[2]>0):
                im.addLink(edge[0], edge[1], edge[2])
            im.add_link(edge[0], edge[1], weight=edge[2])
        else:
            im.add_link(edge[0], edge[1])

    im.run()
    modules=im.get_multilevel_modules()
    modules_items=modules.items()
    membership = [":".join([str(a) for a in membership])
                  for index, membership in im.get_multilevel_modules().items()]

    levelMembership = []
    levelCount = max([len(element.split(":")) for element in membership])
    for level in range(levelCount):
        print(level)
        levelMembership.append([":".join(element.split(":")[:(level+1)]) for element in membership])
        
    return levelMembership

def match_numpy(y,x,arr_exp,bool_value):
    """
    Input values in way:
        y - sub array (find the values)
        x - primary array (where are finding values)
        arr_exp - explored array, chooese values according to array x and y
        bool_value - The way, where are exported values primary finding or supress values from sub array (x) 
        x=good_nodes
        y=papers_ids
        arr_exp=papers_ids
        bool_value=False
    """
    print("\t  Processing_articles ----> def match_numpy()")
    """
    y=papers_ids
    x=xport
    """
    
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)
    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] != y
    mask = x[yindex] != y
    posi = np.ma.array(yindex, mask=mask)
    
    posi_inverse=np.where(mask!=bool_value)[0]
    
    ynew=arr_exp[posi] 
    
    return ynew,posi_inverse

def match_numpy_v2(x,y,arr_exp,bool_value):
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)
    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] != y
    posi = np.ma.array(yindex, mask=mask)
    
    posi_inverse=np.where(mask!=bool_value)[0]
    
    ynew=arr_exp[posi_inverse] 
    
    return ynew,posi_inverse
    

def match4keywords(y,x,arr_exp,bool_value):
    """
    Input values in way:
        y - sub array (find the values)
        x - primary array (where are finding values)
        arr_exp - explored array, chooese values according to array x and y
        bool_value - The way, where are exported values primary finding or supress values from sub array (x) 
        x=good_nodes
        y=papers_ids
        arr_exp=papers_ids
        bool_value=False
    """
    """
    y=papers_ids
    x=xport
    """
    y=keywords_top20
    x=keywords_idx
    arr_exp=keywords_idx_count
    
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)
    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] != y
    mask = x[yindex] != y
    posi = np.ma.array(yindex, mask=mask)
    
    posi_y=posi[posi.mask == False]
    
    posi_y_good=posi_y
    
    keys_idx=x[posi_y]
    keys_values_idx=arr_exp[posi_y]

    return  keys_idx,keys_values_idx

def density_scatter(x, y, name, keywords_intime,keywords,title,bartext , ax = None, sort = True, bins = 50, **kwargs ):
    """
    Scatter plot colored by 2d histogram
    keywords=keyws[::-1]
    """
    print(bins)
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        
    z=(z/np.max(z))
    ax.scatter( x, y, c=cmap(z), **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm,cmap='jet'), ax=ax)
    cbar.mappable.set_clim(vmin=0,vmax=np.max(keywords_intime))

    ax.set_title(title,fontsize=8)
    ax.set_xlabel("Year of publications", fontsize=8)
    ax.set_ylabel("Keywords", fontsize=8)
    ax.xaxis.set_tick_params(labelsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_yticks(np.linspace(start=4-0.5, stop=keywords_intime.shape[0]*4-0.5,num=keywords_intime.shape[0]))
    ax.set_yticklabels(keywords)
    cbar.ax.set_ylabel(bartext,fontsize=8)
    plt.grid(alpha=0.7,linestyle=':')
    fig.savefig("D:/Projekty/2023_Brazilia/bardosova_project-main/v4/"+name+".png", bbox_inches = 'tight',dpi=1000)
    return z,ax


keywords = ["feature dimensionality reduction"]

supp=Support("")
proart=Processing_articles("D:/Clanky/x2025_28_BR_v3_Feature/","xnet/","Feature")
proart.path="D:/Clanky/x2025_28_BR_v3_Feature/"
proart.folder="xnet/"
pb = ProgressBar(0, 1)
pb.finish_value=30
articles_pd = proart.articles_pd
authors_pd = proart.authors_pd
papers_ids=articles_pd["Paper ID"].values
all_keywords=proart.all_keywords
"""
articles_pd=articles_pd.iloc[:1000]
proart.articles_pd=articles_pd
"""
for kw in keywords:
    print(kw)
#proart.make_DB_articles("jaromir.klarak@gmail.com", keywords, 2000000)
clanky= pd.read_json("D:/Clanky/x2024_konfera/xnet/pd_articles.json")
"""Processed"""
#proart.get_articles_citations() #Pending
#proart.get_citing_insideandedges() #Future
#proart.tokenizing() #Processed


v1_forbidden_nodes_A=np.asarray(articles_pd["Paper ID"].values[-2:])
export_name_xnetwork="Feature_v1B"
v1_finded_titles=proart.make_xnetwork(v1_forbidden_nodes_A, export_name_xnetwork,"edges_sim") #stored!!
v1_finded_titles_list=proart.finded_titles 
network_v2 = xn.xnet2igraph(proart.path+export_name_xnetwork+".xnet") 
proart.make_xnetwork_clustered(network_v2)  
reports_tokens_v2=proart.reports_tokens


v2_communitiespre = proart.communitiespre
v2_communities_np=np.asarray(v2_communitiespre).astype(np.int32)
v2_communities_np_uni=np.unique(v2_communities_np,return_counts="True")




############################################################################################################################################


"""
# There was error witn numpy array, need to transform areay to something readable#
article_citing=proart.article_citing
article_citing.to_json(r"D:/Projekty/2023_ECONOMIC/xnet/articles_pd_citations_processed.json")
columns=article_citing.columns.values
cols=[str(columns[0]),str(columns[1]),str(columns[2]),str(columns[3])]
article_citing2=pd.DataFrame(columns=article_citing.columns) 
for idx1 in tqdm(range(len(article_citing))):
    item= article_citing.loc[idx1]
    itemval=item.values
    list_cit=list()
    try:
        for i in range(len(itemval[1])):
            list_cit.append(str(itemval[1][i]))
        item2=np.asarray(item.values[1],dtype="str")
    except:
        list_cit=[str(itemval[1])]

    #item3=pd.DataFrame(columns=cols)
    #item3.loc[0,cols]=[str(itemval[0]),np.asarray(list_cit),int(itemval[2]),int(itemval[3])]
    article_citing2.loc[idx1,cols]=[str(itemval[0]),np.asarray(list_cit),int(itemval[2]),int(itemval[3])]
    if idx1%10000==0:
        try:
            article_citing2.to_json(r"D:/Projekty/2023_ECONOMIC/xnet/articles_pd_citations_processed2.json") 
        except:
            print(f"Do not save: {idx1}")

#article_citing.to_json(r"D:/Projekty/2023_ECONOMIC/xnet/articles_pd_citations_processed.json")
article_citing2.to_json(r"D:/Projekty/2023_ECONOMIC/xnet/articles_pd_citations_processed2.json")
"""
all_keywords=proart.all_keywords

articles_heads=list(articles_pd.columns)
tokens_names=list(all_keywords.columns)
keys_id=all_keywords["tokens_id"].values
keys_ar=articles_pd["Paper ID"].values
tokenized_articles=all_keywords
for article_idx in range(len(keys_id)-1,len(articles_pd)):
    article_id=str(articles_pd.iloc[article_idx][articles_heads[0]])

    
    if article_id not in keys_id:
        article_title=str(articles_pd.iloc[article_idx][articles_heads[1]])
        article_abstract=str(articles_pd.iloc[article_idx][articles_heads[2]])
        
        tokennized_strings,tokennized_strings_di=proart.tokenizeString_v2(article_abstract,30,article_title)
        
        tokenized_articles.loc[article_idx,tokens_names]=[article_id,article_title,article_abstract,tokennized_strings,tokennized_strings_di]
        


tokenized_articles.to_json(proart.path+"pd_articles_keywords_A.json")

#################################################################################################################################################################################
#################################################################################################################################################################################

articles_pd_id=np.asarray(articles_pd["Paper ID"])
years_whole=np.asarray(articles_pd["Year"])
years_whole_uni=np.unique(years_whole,return_counts="True")



years_au_whole_in=np.asarray(authors_pd["Paper ID"].values)
years_au_whole_au=np.asarray(authors_pd["Author ID"].values)
years_au_whole_co=np.asarray(authors_pd["Country"].values)
for idx, item in tqdm(enumerate(years_au_whole_co)):
    if type(item)!=str:
       years_au_whole_co[idx]=np.asarray(["0"]) 

years_au_whole=np.zeros((len(years_au_whole_in)),dtype="int32")
for idx, item in tqdm(enumerate(years_au_whole_in)):
    pdx=np.where(articles_pd_id==item)[0]
    if len(pdx)>0:
        years_au_whole[idx]=years_whole[pdx]


years_au_whole_uni=np.unique(years_au_whole,return_counts="True")  
years_au_whole_co_uni=np.unique(years_au_whole_co,return_counts="True")  
years_authors_npa=np.zeros((len(years_whole_uni[0][64:-2]),3),dtype="float")
for idx, year in tqdm(enumerate(years_whole_uni[0][64:-2])):
    psau=np.where(year==years_au_whole)[0]
    psar=np.where(year==years_whole)[0]
    au_years=years_au_whole_au[psau]
    au_years_uni=np.unique(au_years,return_counts="True")
    years_authors_npa[idx,0]=len(au_years_uni[0])
    years_authors_npa[idx,1]=len(psar)
    years_authors_npa[idx,2]=len(psau)/len(psar)
   
zsortarg=np.argsort(years_au_whole_co_uni[1])[::-1]
country_nn=years_au_whole_co_uni[1][zsortarg]
country_id=years_au_whole_co_uni[0][zsortarg]
years_countries_npa=np.zeros((20,len(years_whole_uni[0][64:-2])),dtype="float")
dict_authors_years={}
for idx, year in tqdm(enumerate(years_whole_uni[0][64:-2])):
    psco=np.where(year==years_au_whole)[0]
    co_years=years_au_whole_co[psco]
    au_years=years_au_whole_au[psco]
    country_year={}
    for zdx, country_zdx in enumerate(country_id[:20]):
        psco_zdx=np.where(country_zdx==co_years)[0]
        au_years_co=au_years[psco_zdx]
        au_years_co_uni=np.unique(au_years_co,return_counts="True")
        years_countries_npa[zdx,idx]=len(psco_zdx)
        





""" PLot all years"""
fig, ax1 = plt.subplots(figsize=(16, 4), dpi=1000)
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax4 = ax1.twinx()

plt.title("Retrieved data from OpenAlex database",fontsize=24,fontweight='bold')

p1, = ax1.plot(years_whole_uni[0][64:-2],years_whole_uni[1][64:-2],c="black",linewidth=2,label = "No. of all articles")
p2, = ax2.plot(years_whole_uni[0][64:-2],years_authors_npa[:,0],c="blue",linewidth=2,label = "No. of authors")
p3, = ax3.plot(years_whole_uni[0][64:-2],years_authors_npa[:,2],c="green",linewidth=2,label = "Average ratio authors per article")
#p4, = ax4.plot(years_whole_uni[0][64:-2],years_authors_npa[:,1]/years_authors_npa[:,0],c="red",linewidth=2,label = "Average article per author")

ax1.set_xlabel("Year",fontsize=22,fontweight='bold')
ax1.xaxis.set_tick_params(labelsize=18)

ax1.yaxis.set_tick_params(labelsize=18)
ax2.yaxis.set_tick_params(labelsize=18)
ax3.yaxis.set_tick_params(labelsize=18)
ax4.yaxis.set_tick_params(labelsize=18)

ax1.set_ylabel("Num. of articles",fontsize=22,fontweight='bold')
ax2.set_ylabel("No. of authors",fontsize=18,fontweight='bold')
ax3.set_ylabel("Ratio authors/article",fontsize=18,fontweight='bold')
ax4.set_ylabel("article per author",fontsize=18,fontweight='bold')

ax1.yaxis.label.set_color(p1.get_color())
ax2.yaxis.label.set_color(p2.get_color())
ax3.yaxis.label.set_color(p3.get_color())
#ax4.yaxis.label.set_color(p4.get_color())

tkw = dict(size=4, width=1.5)
ax2.tick_params(axis='y', colors=p2.get_color(), **tkw)
ax3.tick_params(axis='y', colors=p3.get_color(), **tkw)
#ax4.tick_params(axis='y', colors=p4.get_color(), **tkw)
ax1.tick_params(axis='x', **tkw)

ax3.spines['right'].set_position(('outward', 90))
ax4.spines['right'].set_position(('outward', 140))

ax1.legend(handles=[p1, p2, p3],loc="center left", bbox_to_anchor=(0.92,1.34),fontsize=15)

#ax2.legend(loc="center left", bbox_to_anchor=(1.07,0.73),fontsize=15)
plt.savefig("D:/Clanky/x2025_25_SDS2025_CH/images/number of articlesA.png",bbox_inches='tight', dpi=800)




""" PLot all years"""
fig, ax1 = plt.subplots(figsize=(16, 4), dpi=1000)
change_pap=np.round(((years_whole_uni[1][65:-1]-years_whole_uni[1][64:-2])/years_whole_uni[1][64:-2])*100,2)
change_aut=np.round(((years_authors_npa[1:,0]-years_authors_npa[:-1,0])/years_authors_npa[:-1,0])*100,2)

plt.title("Annual change number of authors and papers",fontsize=24,fontweight='bold')
p1, = ax1.plot(years_whole_uni[0][66:-3],change_pap[:-3],c="black",linewidth=2,label = "Annual change of papers")
p2, = ax1.plot(years_whole_uni[0][66:-3],change_aut[:-2],c="blue",linewidth=2,label = "Annual change of authors")

ax1.set_xlabel("Year",fontsize=22,fontweight='bold')
ax1.yaxis.set_tick_params(labelsize=18)
ax1.set_ylabel("Annual change [%]",fontsize=22,fontweight='bold')

ax1.yaxis.label.set_color(p1.get_color())
tkw = dict(size=4, width=1.5)
ax1.tick_params(axis='x', **tkw)
ax1.legend(handles=[p1, p2],loc="center left", bbox_to_anchor=(0.92,1.2),fontsize=15)
plt.grid(alpha=0.95,linestyle=':')
plt.savefig("D:/Clanky/x2025_25_SDS2025_CH/images/change of articles_v2.png",bbox_inches='tight', dpi=800)





fig, ax = plt.subplots(figsize=(16, 4), dpi=1000)

fruits = list(country_id[:20])
#fruits[39] = "NaN" 
counts = list(country_nn[:20])
bar_labels = ['red', 'blue', '_red', 'orange']
ax.bar(fruits, counts, color="blue",width=0.7)
ax.set_ylabel('Number of authors',fontsize=20,fontweight='bold')
ax.set_xlabel('Country of Affiliation',fontsize=20,fontweight='bold')
ax.set_title('Absolute number of authors according to countries',fontsize=22,fontweight='bold')
ax.xaxis.set_tick_params(labelsize=17)
ax.yaxis.set_tick_params(labelsize=17)

plt.savefig("D:/Clanky/x2025_25_SDS2025_CH/images/nauthors_countries.png",bbox_inches='tight', dpi=800)

"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!need to suprres reeption in authors !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
cmap0=plt.cm.get_cmap('nipy_spectral')
cmap1=plt.cm.get_cmap('cool')
fig, ax1 = plt.subplots(figsize=(16, 4), dpi=100)
plt.title('Number of authors according to country origin (up to 2022)',fontsize=22,fontweight='bold')
for idx in range(10):
    print(idx)
    ax1.plot(years_whole_uni[0][66:-2],years_countries_npa[idx,:-2],c=vars()["cmap"+str(idx%1)]((idx)/9),linewidth=2,label = str(country_id[idx]))

ax1.set_xlabel("Year",fontsize=20,fontweight='bold')
ax1.set_ylabel("Num. of authors",fontsize=20,fontweight='bold')
ax1.xaxis.set_tick_params(labelsize=17)
ax1.yaxis.set_tick_params(labelsize=17)
ax1.legend(loc="center left", bbox_to_anchor=(1.02,0.5),fontsize=15)
plt.savefig("D:/Clanky/x2025_25_SDS2025_CH/images/number of authors2countries_v2.png",bbox_inches='tight', dpi=800)  

#################################################################################################################################################################################
#################################################################################################################################################################################
""" Population of clusters """



"""PLotting rise of clusters in time"""

# data from United Nations World Population Prospects (Revision 2019)
# https://population.un.org/wpp/, license: CC BY 3.0 IGO


clusters_np=list()
for idx, uni_idx in enumerate(v3_communities_np_uni[0]):
    clusters_positions=np.where(uni_idx==v3_communities_np)[0]
    if idx==0:
        clusters_np1=clusters_positions
    else:
        clusters_np1=np.concatenate((clusters_np1,clusters_positions),axis=0)
    
    clusters_np.append(clusters_positions)

test=np.unique(clusters_np1,return_counts="True")

nodes_np=np.asarray(network_v3.vs['wos_id'])
names   =np.asarray(network_v3.vs["name"])

nodes_sub=nodes_np[clusters_np1]
names_sub=names[clusters_np1]



articles_pd_id=np.asarray(articles_pd["Paper ID"])
test,posi_id=match_numpy(articles_pd_id,nodes_sub,articles_pd_id,True)

articles_pd_clustered=articles_pd.loc[posi_id]
all_keywords_clustered=all_keywords.loc[posi_id]

keywords_uni_clusters=proart.get_uni_values(all_keywords_clustered)
keywords_uni=proart.get_uni_values(all_keywords)
years_all=np.asarray(articles_pd["Year"])
years_all_uni=np.unique(years_all,return_counts="True")
years_all_clustered=np.asarray(articles_pd_clustered["Year"])
years_all_uni_clustered=np.unique(years_all_clustered,return_counts="True")

years_1920_index=np.where((years_all_uni_clustered[0]>1970)&(years_all_uni_clustered[0]<2023))[0]
years_1920=years_all_uni[1][years_1920_index]
keywords_top20 = np.asarray(["anomaly", "defect", "detection"])

keywords_top20_num=np.zeros((len(keywords_top20), len(years_1920)),dtype="int")
dict_keyword_years=pd.DataFrame(index=list(keywords_top20))
network_v5_cluster_index_top10=np.asarray(network_v3.vs["Cluster Index"],dtype="int32")
network_v5_cluster_index_top10_uni=np.unique(network_v5_cluster_index_top10,return_counts="true")
network_v5_cluster_index_top10_size=np.zeros((len(network_v5_cluster_index_top10_uni[0]),len(years_1920_index)),dtype="int32")
for idx, year in enumerate(years_all_uni_clustered[0][years_1920_index]):
    #year=years_whole_uni[0][years_1920][idx]
    posi_idx=np.where(years_all_clustered==year)[0]
    keywords_year=proart.get_uni_values(all_keywords_clustered.iloc[posi_idx])
    keywords_idx=np.asarray(keywords_year.index).astype(str)
    keywords_idx_count=np.asarray(keywords_year)
    
    for item in keywords_top20:
        item_count=0
        for idx2 in range(len(keywords_year)):
            if item in str(keywords_idx[idx2]) or item ==str(keywords_idx[idx2]):
                item_count+=keywords_idx_count[idx2]
        dict_keyword_years.loc[item, year]=item_count
    
    array_cls=network_v5_cluster_index_top10[posi_idx]
    for zdx,cls1 in enumerate(network_v5_cluster_index_top10_uni[0]):
        clsps=np.where(cls1==array_cls)[0]
        network_v5_cluster_index_top10_size[zdx,idx]=len(clsps)





cls_years = years_all_uni_clustered[0][years_1920_index]
population_clusters = {}
population_clusters1 = {}
population_clusters2 = {}
population_clusters3 = {}
network_cluster=np.asarray(network_v3.vs["Cluster Index"],dtype="int32")
network_cluster_uni=np.unique(network_cluster,return_counts="True")
net5_top10_index=np.asarray(network_cluster_uni[0],dtype="object")

sum_of=np.sum(network_v5_cluster_index_top10_size,axis=0)
sum_of1=np.sum(network_v5_cluster_index_top10_size[1:,:],axis=0)
names=["A","B","C"]
for idx,item in enumerate(net5_top10_index):
    population_clusters[names[idx]]=network_v5_cluster_index_top10_size[idx,:]
    population_clusters2[names[idx]]=network_v5_cluster_index_top10_size[idx,:]/sum_of*100
    print(np.sum(network_v5_cluster_index_top10_size[idx,:]))
    if idx==0:
        ztz=np.zeros(len(network_v5_cluster_index_top10_size[idx,:]),dtype="int32")
        population_clusters1[names[idx]]=ztz
        population_clusters3[names[idx]]=ztz
    else:
        population_clusters1[names[idx]]=network_v5_cluster_index_top10_size[idx,:]  
        population_clusters3[names[idx]]=network_v5_cluster_index_top10_size[idx,:]/sum_of1*100
        
fig, ax = plt.subplots(2, 2, figsize=(16, 9),constrained_layout=True)
ax[0][0].stackplot(cls_years, population_clusters.values(),
             labels=population_clusters.keys(), alpha=0.8)
ax[0][0].legend(loc='upper left',fontsize=15)
ax[0][0].set_title('Development size of clusters',fontsize=20,fontweight='bold')
ax[0][0].set_xlabel('Year',fontsize=17,fontweight='bold')
ax[0][0].set_ylabel('Number of articles\n in clusters',fontsize=17,fontweight='bold')

ax[0][1].stackplot(cls_years, population_clusters1.values(),
             labels=population_clusters1.keys(), alpha=0.8)
ax[0][1].legend(loc='upper left',fontsize=15)
ax[0][1].set_title('Development size of clusters without "A"',fontsize=20,fontweight='bold')
ax[0][1].set_xlabel('Year',fontsize=17,fontweight='bold')
ax[0][1].set_ylabel('Number of articles in clusters',fontsize=17,fontweight='bold')

ax[1][0].stackplot(cls_years, population_clusters2.values(),
             labels=population_clusters2.keys(), alpha=0.8)
ax[1][0].legend(loc='upper left',fontsize=15)
ax[1][0].set_title('Development relative size of clusters',fontsize=20,fontweight='bold')
ax[1][0].set_xlabel('Year',fontsize=17,fontweight='bold')
ax[1][0].set_ylabel('Percent share of articles\n in clusters [%]',fontsize=17,fontweight='bold')

ax[1][1].stackplot(cls_years, population_clusters3.values(),
             labels=population_clusters3.keys(), alpha=0.8)
ax[1][1].legend(loc='upper left',fontsize=15)
ax[1][1].set_title('Development relative size of clusters without "A"',fontsize=20,fontweight='bold')
ax[1][1].set_xlabel('Year',fontsize=17,fontweight='bold')
ax[1][1].set_ylabel('Percent share of articles\n in clusters [%]',fontsize=17,fontweight='bold')

ax[0][0].yaxis.set_tick_params(labelsize=15)
ax[0][1].yaxis.set_tick_params(labelsize=15)
ax[1][0].yaxis.set_tick_params(labelsize=15)
ax[1][1].yaxis.set_tick_params(labelsize=15)
ax[0][0].xaxis.set_tick_params(labelsize=15)
ax[0][1].xaxis.set_tick_params(labelsize=15)
ax[1][0].xaxis.set_tick_params(labelsize=15)
ax[1][1].xaxis.set_tick_params(labelsize=15)

plt.savefig("D:/Clanky/x2025_25_SDS2025_CH/images/net3_clusters.png",bbox_inches="tight",dpi=1000)


#######################################################################################################################################
"""Reduce edges in netv2 and met_v3A"""
network_v3_c = xn.xnet2igraph("D:/Clanky/x2024_konfera/xnet/defect_v2A_clustered1A.xnet") 
v3c_wos=np.asarray(network_v3_c.vs['wos_id'])
v3c_name=np.asarray(network_v3_c.vs["name"])
v3c_cluster_index=np.asarray(network_v3_c.vs["Cluster Index"])
v3c_cluster_name=np.asarray(network_v3_c.vs["Cluster Name"])
v3c_cluster_title=np.asarray(network_v3_c.vs["title"])
edges_v3c=np.asarray([(e.source,e.target) for e in network_v3_c.es])

irn=np.sort(np.asarray(random.sample(list(np.arange(len(edges_v3c))), int(len(edges_v3c)*0.15)),dtype="int32"))



netc = xn.xnet2igraph("D:/Clanky/x2024_konfera/xnet/defect_v3C_clustered1A.xnet") 
clus=np.unique(np.asarray(netc.vs["Cluster Name"]),return_counts="True")
print(netc.vcount())
print(netc.ecount())
print(len(clus[0]))


new_edges=edges_v3c[irn]

g = ig.Graph(directed=True)
g.add_vertices(v3c_wos)
g.add_edges(new_edges)
g.vs['name'] = v3c_name
g.vs['wos_id'] = v3c_wos
g.vs['Cluster Index'] = v3c_cluster_index
g.vs['Cluster Name'] = v3c_cluster_name
g.vs['title'] = v3c_cluster_title

print(g.vcount())
print(g.ecount())

xn.igraph2xnet(g, "D:/Clanky/x2024_konfera/xnet/defect_v2A_clustered1A_R.xnet")


###################################################################################################################################################################
"""Printing keywords from net_v2A"""
for idx, item in enumerate(list(reports_tokens_v2.keys())[:10]):
    sub_dict=reports_tokens_v2[item]
    list_key=list(sub_dict.keys())
    ddi=sub_dict.sort_values(list_key[-2],ascending=False)[0:200] 
    ddiv=ddi.values
    msg=""
    if len(ddiv)>200:
        range_val=200
    else:    
        range_val=len(ddiv)
    
    for i in range(range_val):
        msg+=str(ddiv[i,0])+": "+str(np.round(ddiv[i,8],3))+"; "
        
    print(msg+"\n")
        




###########################################################################################################################################################################################
###########################################################################################################################################################################################

import v
import os
import transformers
import openai


os.environ["OPENAI_API_KEY"] = "KEY" #need to use won key




def generate_content(gpt_assistant_prompt: str, gpt_user_prompt: str) -> dict:
    messages = [
        {"role": "assistant", "content": gpt_assistant_prompt},
        {"role": "user", "content": gpt_user_prompt}
    ]
    response = openai.ChatCompletion.create(model="gpt-4o-mini",messages=messages,temperature=0.7)
    return response["choices"][0]["message"]["content"].strip()

gpt_assistant_prompt = ''

list_of_titles=list()
list_of_titles_keys=list()
for clusterIndex in range(len(v2_communities_np_uni[0][:10])):
    
    names_columns=list(reports_tokens_v4B[clusterIndex].columns )
    pd_cluster_reports1=reports_tokens_v4B[clusterIndex].sort_values(by=[names_columns[8]],ascending=False)[:100] 
    msg_keys=""
    msg_keys_only=""
    list_keys=list(pd_cluster_reports1[names_columns[0]])
    list_vals=list(pd_cluster_reports1[names_columns[8]])
    
    if len(list_keys)>100:
        ilen=100
    else:
        ilen=len(list_keys)
    for i in range(ilen):
        msg_keys+=str(list_keys[i])+", "+str(np.round(list_vals[i],4))+", "
        msg_keys_only+=str(list_keys[i])+", "
    
    
    msg_pre="""I will provide you a list with entries in the form: keyword + number. Each entry in the list contains a term representative of a research topic followed by a number that gives a measure of the importance of the term for the topic. 
    Notice that all terms have been extracted from the titles and abstracts of scientific papers retrieved with the following query ("defect" AND "detection") OR ("detection" AND "anomaly").
    Please, output an informative descriptive title for the topic. Be as specific as possible and avoid using generic qualifying words such as `advanced´, ´innovative' , ´comprehensive':"""
    
    if True:
        "OpenAI"
        try:
            result = generate_content(gpt_assistant_prompt, msg_pre+msg_keys)
            print(result)
            
        except:
            clust_title="Error in: "+str(clusterIndex)
            
    list_of_titles_keys.append(str(msg_keys_only))        
    list_of_titles.append(str(result))


list_of_titles_s=list_of_titles
list_of_titles=list()
for idx, item in enumerate(list_of_titles_s):
    item=item.replace("*","")
    item=item.replace("Title","")
    item=item.replace(": ","")
    item=item.replace('"',"")
    list_of_titles.append(item)
    
    
network_v3_b = xn.xnet2igraph("D:/Clanky/x2024_konfera/xnet/defect_v3B_clustered1A.xnet") 

v3b_wos=np.asarray(network_v3_b.vs['wos_id'])
v3b_name=np.asarray(network_v3_b.vs["name"])
v3b_cluster_index=np.asarray(network_v3_b.vs["Cluster Index"],dtype="int32")
v3b_cluster_name=np.asarray(network_v3_b.vs["Cluster Name"])
v3b_cluster_title=np.asarray(network_v3_b.vs["title"])
edges_v3b=np.asarray([(e.source,e.target) for e in network_v3_b.es])

v3b_ciuni=np.unique(v3b_cluster_index,return_counts="True")

node_titles=list()
v3b_cluster_nameB=list()
titles=list()
for idx, clu in enumerate(v3b_cluster_index):
    node_titles.append(str(clu)+": "+list_of_titles[int(clu)])
    v3b_cluster_nameB.append(str(clu)+": "+str(v3b_cluster_name[idx]).split("-")[1][1:])



g = ig.Graph(directed=True)
g.add_vertices(v3b_wos)
g.add_edges(edges_v3b)
g.vs['name'] = v3b_name
g.vs['wos_id'] = v3b_wos
g.vs['Cluster Index'] = v3b_cluster_index
g.vs['Cluster keywords'] = v3b_cluster_nameB
g.vs['Cluster title'] = node_titles


print(g.vcount())
print(g.ecount())

xn.igraph2xnet(g, "D:/Clanky/x2024_konfera/xnet/defect_v3B_v4r.xnet")

####################################################################################################################################################################


clusters_np=list()
for idx, uni_idx in enumerate(v3b_ciuni[0][ddi_indexes]):
    clusters_positions=np.where(uni_idx==v3b_cluster_index)[0]
    if idx==0:
        clusters_np1=clusters_positions
    else:
        clusters_np1=np.concatenate((clusters_np1,clusters_positions),axis=0)
    
    clusters_np.append(clusters_positions)

test=np.unique(clusters_np1,return_counts="True")


test,posi_cc=match_numpy(edges[:,0],clusters_np1,edges[:,0],False)
test,posi_cc1=match_numpy(edges[:,1],clusters_np1,edges[:,1],False)

#posi_c2=np.unique((np.concatenate((posi_cc,posi_cc1),axis=0)),return_counts="True")
#posi_c2A=posi_c2[0]
posi_c2=np.intersect1d(posi_cc, posi_cc1)
posi_c2A=posi_c2

edges_sub=edges[posi_c2A,:]
edges_sub_uni=np.unique(edges_sub,return_counts="True")[0]

edges_sub1=list()
for item in edges_v3b:
    """ and  ---  or : for detemrined gray area in sub network with specific clusters  """
    if item[1] in clusters_np1 and item[0] in clusters_np1:
        edges_sub1.append([v3b_wos[item[0]],v3b_wos[item[1]]])

edges_sub1_np=np.asarray(edges_sub1)    
edges_sub_uni=np.unique(edges_sub1_np,return_counts="True")
edges_sub=np.copy(edges_sub1_np) 



v3bf_wos=list()
v3bf_name=list()
v3bf_cluster_index=list()
v3bf_cluster_nameB=list()
nodef_titles=list()
for idx, item in enumerate(v3b_cluster_index):
    if item in ddi_indexes:
       v3bf_wos.append(v3b_wos[idx])
       v3bf_name.append(v3b_name[idx])
       v3bf_cluster_index.append(v3b_cluster_index[idx])
       v3bf_cluster_nameB.append(v3b_cluster_nameB[idx])
       nodef_titles.append(node_titles[idx])
        


g = ig.Graph(directed=True)
g.add_vertices(v3bf_wos)
g.add_edges(edges_sub1)
g.vs['name'] = v3bf_name
g.vs['wos_id'] = v3bf_wos
g.vs['Cluster Index'] = v3bf_cluster_index
g.vs['Cluster keywords'] = v3bf_cluster_nameB
g.vs['Cluster title'] = nodef_titles


print(g.vcount())
print(g.ecount())

xn.igraph2xnet(g, "D:/Clanky/x2024_konfera/xnet/defect_v3B_FOCUSR_Rev3.xnet")

reports_tokens_v4B_focused=reports_tokens_v4B[0]


ztitles=np.unique(np.asarray(nodef_titles),return_counts="True")
arg_sort=np.argsort(ztitles[1])[::-1]
for idx, item in enumerate(ztitles[0][arg_sort]):
    print(item.split(":")[0]+"\t"+str(ztitles[1][arg_sort][idx])+"\t"+item.split(":")[1][:])


##########################################################################################################################################################################################
""" WORLD MAP """
##########################################################################################################################################################################################


institut_col=["Institution ID","display_name","city","country","country_code","lat","long","author_num","works_count","cited_by_count","counts_by_year","ratio_WA"]
ps=np.where(np.asarray(v3bf_cluster_index,dtype="int32")==0)[0]
papers_c0=np.asarray(v3bf_wos)[ps]
papers_ca=np.asarray(authors_pd['Paper ID'])
psa=match_numpy(papers_ca, papers_c0, papers_ca, True)[1]

institut_uni=np.unique(authors_pd["Institution ID"].values[psa],return_counts="True")

db_institutions=pd.DataFrame(columns=institut_col)
institut_col1=list(db_institutions.columns)
for pdx,inst_id in tqdm(enumerate(institut_uni[0])):
    url="https://api.openalex.org/institutions/"+str(inst_id)
    page_with_results = requests.get(url).json()
    
    instgeo=page_with_results["geo"]
    test=[inst_id ,page_with_results["display_name"], page_with_results["geo"]["city"], page_with_results["geo"]["country"], 
                                       page_with_results["geo"]["country_code"] ,  page_with_results["geo"]["latitude"], page_with_results["geo"]["longitude"], institut_uni[1][pdx] ,
                                       page_with_results["works_count"] , page_with_results["cited_by_count"], page_with_results["counts_by_year"] , float(page_with_results["works_count"]/institut_uni[1][pdx])]
    db_institutions.loc[int(pdx),institut_col1]=test

db_institutions.to_json(r"D:/Clanky/x2024_konfera/xnet/pd_institutions.json")  

authorswm=pd.read_json("D:/Clanky/x2024_konfera/xnet/pd_institutions.json")
authorswm["quality"]=(authorswm["cited_by_count"]/authorswm["works_count"])

####################################################
authors_pdc1=authors_pd.values[psa,:]
c1_num_author=np.zeros(len(authorswm),dtype="int32")
c1_num_works=np.zeros(len(authorswm),dtype="int32")
c1_num_works_metric=np.zeros((len(authorswm),3),dtype="float")

aut_int=authors_pdc1[:,14]
aut_aut=authors_pdc1[:,3]
aut_wor=authors_pdc1[:,0]

articles_np=articles_pd.values[:,[0,5]]
for pdx, institut in tqdm(enumerate(authorswm["Institution ID"].values)):
    institut=authorswm["Institution ID"].values[pdx]
    posi=np.where(np.asarray([institut])==aut_int)[0]
    uni_aut=np.unique(aut_aut[posi],return_counts="True")
    uni_wor=np.unique(aut_wor[posi],return_counts="True")
    if len(uni_aut[0])>0:
        c1_num_author[pdx]=len(uni_aut[0])

    if len(uni_wor[0])>0:    
        c1_num_works[pdx]=len(uni_wor[0])
        c1_num_works_metric[pdx,0]=np.mean(uni_wor[1])
        test,wps=match_numpy(articles_np[:,0], uni_wor[0], articles_np[:,0], True)
        c1_num_works_metric[pdx,1]=np.sum(articles_np[wps,1])


clusters_np=list()
for idx, uni_idx in enumerate([v3b_ciuni[0][0]]):
    clusters_positions=np.where(uni_idx==v3b_cluster_index)[0]
    if idx==0:
        clusters_np1=clusters_positions
    else:
        clusters_np1=np.concatenate((clusters_np1,clusters_positions),axis=0)
    


good_edges=list()
for item in edges_v3b:
    """ and  ---  or : for detemrined gray area in sub network with specific clusters  """
    if item[1] in clusters_np1 and item[0] in clusters_np1:
        good_edges.append([v3b_wos[item[0]],v3b_wos[item[1]]])




good_edges=np.asarray(good_edges)
authors_np=authors_pdc1[:,np.asarray([0,3,6])]
num_edges=np.zeros((len(good_edges),3),dtype="int")
for idx in tqdm(range(len(good_edges))):
    ps1=np.where(good_edges[idx,0]==authors_np[:,0])[0]
    ps2=np.where(good_edges[idx,1]==authors_np[:,0])[0]
    num_edges[idx,:]=np.asarray([len(ps1),len(ps2),0])

num_edges[:,2]=num_edges[:,0]*num_edges[:,1]

connections=np.zeros((np.sum(num_edges[:,2])+len(num_edges[num_edges[:,2]==0,2]),7),dtype="object")
pdx=0
for idx in tqdm(range(len(good_edges))):
    ps1=np.where(good_edges[idx,0]==authors_np[:,0])[0]
    ps2=np.where(good_edges[idx,1]==authors_np[:,0])[0]
    if len(ps1)>0 and len(ps2)>0:
        for it1 in ps1:
            for it2 in ps2:
                connections[pdx,:2]= np.asarray([authors_np[it1,2],authors_np[it2,2]])
                connections[pdx,6]= 1
                pdx+=1


institutes_np=authorswm.values   
         
for pdx2, institut in tqdm(enumerate(authorswm["Institution ID"].values)):
    ps1=np.where(np.asarray([institut])==connections[:,0])[0]
    ps2=np.where(np.asarray([institut])==connections[:,1])[0]
    connections[ps1,2:4]=institutes_np[pdx2,5:7]
    connections[ps2,4:6]=institutes_np[pdx2,5:7]

connections_copy=np.copy(connections)


test=connections[:,2:6]
posi_zeros=np.where(test==0)

posi_zeros_non=np.setdiff1d(np.arange(len(connections)), posi_zeros[0])
connections=connections[posi_zeros_non]

connections_pd=pd.DataFrame({'lat_source':np.asarray(connections[:,2],dtype="float"),'long_source':np.asarray(connections[:,3],dtype="float"),
                             'lat_destination':np.asarray(connections[:,4],dtype="float"),'long_destination':np.asarray(connections[:,5],dtype="float"),
                             "cit_quality":(np.asarray(connections[:,6],dtype="float")-np.min(np.asarray(connections[:,6],dtype="float")))})
authorswm["quality"]=(authorswm["cited_by_count"]/authorswm["works_count"])
connections_pd=connections_pd.dropna()

con_dic={"connections_pd":connections_pd,"connections":connections,"num_edges":num_edges,"c1_num_author":c1_num_author,"c1_num_works":c1_num_works}
supp.save_dict("", con_dic, "world_connections_sciBERT")


con_dic=supp.load_dict("", "world_connections_sciBERT")
connections=con_dic["connections"]
conn1=connections[:,0]
for pdx, item in enumerate(conn1):
    if type(item)!=str:
        print(item)
        conn1[pdx]="empty"
con_uni=np.unique(conn1,return_counts="True")

connections_v2=np.asarray([[],[],[],[],[],[],[],[],[],[]]).T
num=0
institutions_ID=authorswm["Institution ID"].values
authorswm_display_name=authorswm["display_name"].values
for pdx, institut in tqdm(enumerate(con_uni[0])):
    ps1=np.where(institut==conn1)[0]
    connsub=connections[ps1,:]
    psi1=np.where(institutions_ID==institut)[0]
    connsubuni=np.unique(connsub[:,1], return_counts="True")
    num+=len(connsubuni[0])
    for pdx2, institut2 in enumerate(connsubuni[0]):
        ps2=np.where(institut2 ==connsub[:,1])[0]
        psi2=np.where(institutions_ID==institut2)[0]
        connections_pdx2=np.asarray([[str(authorswm_display_name[psi1]),str(authorswm_display_name[psi2]),institut,institut2,connections[ps1[0],2],connections[ps1[0],3],connsub[ps2[0],4],connsub[ps2[0],5],len(ps2),np.mean(connsub[ps2[0],6])]],dtype="object")
        connections_v2=np.concatenate((connections_v2,connections_pdx2),axis=0)




conne_posi1=np.where((connections_v2[:,0]!=4)&(connections_v2[:,5]!=0)&(connections_v2[:,6]!=0)&(connections_v2[:,7]!=0)&(connections_v2[:,8]>20))[0]
connections_v2A=connections_v2[conne_posi1]

consim2=np.asarray(connections_v2A[:,9],dtype="float")
consim2A=consim2-np.min(consim2)
consim2B=consim2A/np.max(consim2A)

conne_posi1=np.where((connections_v2[:,0]!=4)&(connections_v2[:,5]!=0)&(connections_v2[:,6]!=0)&(connections_v2[:,7]!=0)&(connections_v2[:,8]>20))[0]
connections_v2A=connections_v2[conne_posi1]
connections_pd2=pd.DataFrame({'lat_source':np.asarray(connections_v2A[:,4],dtype="float"),'long_source':np.asarray(connections_v2A[:,5],dtype="float"),
                             'lat_destination':np.asarray(connections_v2A[:,6],dtype="float"),'long_destination':np.asarray(connections_v2A[:,7],dtype="float"),
                             'mark_size':np.asarray(connections_v2A[:,8],dtype="float"),'mark_color':np.asarray(consim2B,dtype="float")})


import os
import matplotlib.pyplot as plt
from netCDF4 import Dataset as netcdf_dataset
import numpy as np
import cartopy
from cartopy import config
import cartopy.crs as ccrs
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import math
from shapely.geometry import LineString


institution_report=authorswm
c1_num_author=np.zeros(len(authorswm),dtype="int32")
c1_num_works=np.zeros(len(authorswm),dtype="int32")
c1_num_works_metric=np.zeros((len(authorswm),3),dtype="float")

aut_int=authors_pdc1[:,6]
aut_aut=authors_pdc1[:,3]
aut_wor=authors_pdc1[:,0]

articles_np=articles_pd.values[:,[0,5]]
for pdx, institut in tqdm(enumerate(authorswm["Institution ID"].values)):
    institut=authorswm["Institution ID"].values[pdx]
    posi=np.where(np.asarray([institut])==aut_int)[0]
    uni_aut=np.unique(aut_aut[posi],return_counts="True")
    uni_wor=np.unique(aut_wor[posi],return_counts="True")
    if len(uni_aut[0])>0:
        c1_num_author[pdx]=len(uni_aut[0])

    if len(uni_wor[0])>0:    
        c1_num_works[pdx]=len(uni_wor[0])
        c1_num_works_metric[pdx,0]=np.mean(uni_wor[1])
        test,wps=match_numpy(articles_np[:,0], uni_wor[0], articles_np[:,0], True)
        c1_num_works_metric[pdx,1]=np.sum(articles_np[wps,1])
        
    
c1_num_works_metric[c1_num_author==0,0]=0

institution_report["Num authors"]=c1_num_author
institution_report["Num papers"]=c1_num_works
institution_report["Num a/p"]=c1_num_works_metric[:,0]
institution_report["Num cit"]=c1_num_works_metric[:,1]
a=c1_num_works_metric[:,1]/c1_num_author
a[c1_num_author==0]=0
b=c1_num_works_metric[:,1]/c1_num_works
b[c1_num_works==0]=0
institution_report["Num cit/a"]=a
institution_report["Num cit/p"]=b


item_metric=institution_report["Num papers"]
lognp=np.zeros(len(item_metric),dtype="float")
for idx in range(len(item_metric)):
    if (item_metric.values[idx])>1:
        lognp[idx]=math.log(int(np.max(item_metric)),(item_metric.values[idx]))
    else:
        lognp[idx]=0
institution_report["quality log"]=lognp
loginv=1/lognp
loginv[lognp==0]=0
institution_report["quality loginv"]=loginv/np.max(loginv)

cmap=plt.cm.get_cmap('jet')
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson()}, figsize=(20,20))
ax.patch.set_facecolor('black')
ax.patch.set_edgecolor('blue')
ax.add_feature(cartopy.feature.BORDERS, linestyle=':', alpha=1)
ax.add_feature(cartopy.feature.OCEAN, facecolor=("lightblue"))
ax.add_feature(cartopy.feature.LAND)
ax.coastlines(resolution='10m')

indexes_int=institution_report[institution_report["Num papers"]>-1]
indexes_aut=authorswm[institution_report["Num papers"]>-1]    
geometry_air = [Point(xy) for xy in zip(indexes_aut['long'], indexes_aut['lat'])]
gdf =  gpd.GeoDataFrame(indexes_aut, geometry=geometry_air,crs='EPSG:4326')    

gdf.plot(ax=ax,transform=ccrs.Geodetic(),c=cmap(indexes_aut["quality loginv"]),markersize=indexes_int["Num authors"]/8) #c=cmap

#routes_geometry = [LineString([[connections_pd2.iloc[i]['long_source'], connections_pd2.iloc[i]['lat_source']], [connections_pd2.iloc[i]['long_destination'], connections_pd2.iloc[i]['lat_destination']]]) for i in range(connections_pd2.shape[0])]
#routes_geodata = gpd.GeoDataFrame(connections_pd2, geometry=routes_geometry, crs='EPSG:4326') 
#routes_geodata.plot(ax=ax, transform=ccrs.Geodetic(), color=cmap(connections_pd2['mark_color']/np.max(connections_pd2['mark_color'])), linewidth=0.1, alpha=0.08)
#routes_geodata.plot(ax=ax, transform=ccrs.Geodetic(), color="blue", linewidth=0.3, alpha=0.15)

plt.savefig("D:/Clanky/x2025_25_SDS2025_CH/images/worldmap_defect_V3.png",dpi=1000)
    
    
    
    
    
    
v2_communitiespre  =network_v2.vs["Cluster Index"]  
v2_communities_np=np.asarray(v2_communitiespre).astype(np.int32)
v2_communities_np_uni=np.unique(v2_communities_np,return_counts="True")

v2_communities_np1=proart.communities_np
v2_communities_np_uni1=np.unique(v2_communities_np1,return_counts="True")

for clusterIndex in range(len(v2_communities_np_uni[0][:10])):
    
    names_columns=list(reports_tokens_v2[clusterIndex].columns )
    pd_cluster_reports1=reports_tokens_v2[clusterIndex].sort_values(by=[names_columns[8]],ascending=False)[:100] 
    msg_keys=""
    msg_keys_only=""
    list_keys=list(pd_cluster_reports1[names_columns[0]])
    list_vals=list(pd_cluster_reports1[names_columns[8]])
    
    if len(list_keys)>100:
        ilen=100
    else:
        ilen=len(list_keys)
    for i in range(ilen):
        msg_keys+=str(list_keys[i])+", "+str(np.round(list_vals[i],5))+", "
        msg_keys_only+=str(list_keys[i])+", "
    
    print(msg_keys)
    
    
#####################################################################################
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer,  AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity


pretrained_model = 'allenai/scibert_scivocab_uncased'
sciBERT_tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=True)

model = AutoModelForSequenceClassification.from_pretrained(pretrained_model,
                                                          output_attentions=False,
                                                          output_hidden_states=True)

def convert_single_abstract_to_embedding(tokenizer, model, in_text, MAX_LEN = 510):
    """
    tokenizer=sciBERT_tokenizer
    model=model
    in_text=query_text
    MAX_LEN = 510   
    """
    input_ids = tokenizer.encode(
                        in_text, 
                        add_special_tokens = True, 
                        max_length = MAX_LEN, truncation=True                          
                   )    

    results = pad_sequences([input_ids], maxlen=MAX_LEN, dtype="long", 
                              truncating="post", padding="post")  #truncation=True
    
    # Remove the outer list.
    input_ids = results[0]

    # Create attention masks    
    attention_mask = [int(i>0) for i in input_ids]
    
    # Convert to tensors.
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)

    # Add an extra dimension for the "batch" (even though there is only one 
    # input in this batch.)
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    #input_ids = input_ids.to(device)
    #attention_mask = attention_mask.to(device)
    
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    with torch.no_grad():        
        logits, encoded_layers = model(
                                    input_ids = input_ids, 
                                    token_type_ids = None,
                                    #truncation=True,
                                    attention_mask = attention_mask,
                                    return_dict=False)

    layer_i = 12 # The last BERT layer before the classifier.
    batch_i = 0 # Only one input in the batch.
    token_i = 0 # The first token, corresponding to [CLS]
        
    # Extract the embedding.
    embedding = encoded_layers[layer_i][batch_i][token_i]

    # Move to the CPU and convert to numpy ndarray.
    embedding = embedding.detach().cpu().numpy()

    return(embedding)

def process_query(query_text):
    """
    # Create a vector for given query and adjust it for cosine similarity search
    """
    query_vect = convert_single_abstract_to_embedding(sciBERT_tokenizer, model, query_text)
    query_vect = np.array(query_vect)
    query_vect = query_vect.reshape(1, -1)
    return query_vect

nodes_pre=np.asarray(articles_pd["Paper ID"])
titles=np.asarray(articles_pd["Title"])
abstracts=np.asarray(articles_pd["Abstract"])
nodes=list()
for idx in range(len(nodes_pre)):
    nodes.append(nodes_pre[idx])
nodes=np.asarray(nodes)

gnodes=list()
gabstr=list()
gtitle=list()
for idx in range(len(nodes)):
    if ("RETURN TO ISSUE" not in abstracts[idx]) and ("RETURN TO ARTICLES" not in abstracts[idx]):
        gnodes.append(str(nodes[idx]))
        gabstr.append(abstracts[idx])
        gtitle.append(titles[idx])
    



query_vect_list=list()
for idx in tqdm(range(len(gnodes))):    
    query_vect_list.append(process_query(str(gabstr[idx])))
    
edges_cos_sim=np.zeros((int(len(gnodes)*(len(gnodes)-1)/2),1),dtype="float")
edges_cos=np.zeros((int(len(gnodes)*(len(gnodes)-1)/2),2),dtype="object")
num_i=0
print("Cosine similaritiey")
for idx in tqdm(range(len(gnodes)-1)):
    for idx2 in range(idx+1,len(gnodes)):
        cos_sim=float(cosine_similarity(query_vect_list[idx], query_vect_list[idx2])[0, 0]) 
        edges_cos_sim[num_i,0]=cos_sim
        
        edges_cos[num_i,0]=gnodes[idx]
        edges_cos[num_i,1]=gnodes[idx2]
        num_i+=1
        

ps0=np.where(edges_cos_sim[:,0]==0)[0]
edges_uni=np.unique(np.round(edges_cos_sim[:,0],3),return_counts="True")
  
pstop=np.where(((edges_cos_sim[:,0]>0.93) & (edges_cos_sim[:,0]<0.98)) | ((edges_cos_sim[:,0]>0.6495) & (edges_cos_sim[:,0]<0.65)))[0]
    
pd_edges=pd.DataFrame({"0":edges_cos[pstop,0],"1":edges_cos[pstop,1]})
pd_edges.to_json(r""+proart.path+proart.folder+"edges_sim.json") 
    

fig = plt.figure(figsize = (8,5))
plt.plot(edges_uni[0][2:-12],edges_uni[1][2:-12])   
plt.title("Cosine similarity",fontsize=18)
plt.ylabel("Number of similarities",fontsize=15)
plt.xlabel("Similarity value",fontsize=15) 
plt.savefig("D:/Clanky/x2025_28_BR_v3_Feature/images/cosim.png",bbox_inches = 'tight',dpi=1000 )   
    
    
    




