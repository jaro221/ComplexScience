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
    

class Data_Processing():
    """ Need to fullfil missing moduls """
    def __init__(self):
        pass
    
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
            print("Loaded keywords from json saved file to tokenizing abstracts by -Data_gathering.tokenizing-")
        except:
            tokenized_articles=pd.DataFrame(columns=["tokens_id","tokens_titles","tokens_abstracts","tokens_article","tokens_distance"])
            print("Initialized empty datagrame for keywords to tokenizing abstracts by -Data_gathering.tokenizing-")
            

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
        print("Data_gathering----> def make_xnetwork()")
        
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
        xn.igraph2xnet(g, self.path+self.folder+"network_veri2.xnet")

        network = xn.xnet2igraph(self.path+self.folder+"network_veri2.xnet")
        network.vs['wos_id'] = network.vs['name']
        network.vs['name'] = network.vs['title']

        xn.igraph2xnet(network,self.path+self.folder+str(export_name_xnetwork)+".xnet",ignoredNodeAtts=["Text"])
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
        print("\t  Data_gathering ----> def match_numpy()")
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
        print("Data_gathering ----> def get_uni_values()")
        
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
        print(f"\nData_gathering ----> def get_uni_values()  Initiliazed large numpy array     Time: {time.time() - start_time} sec.\n")
        
        start_time=time.time()
        for idx,item in enumerate(columns):
            if idx==0:
                pd_new=new_numpy_pd[item].copy()
            else:
                pd_new=pd.concat([pd_new, new_numpy_pd[item]])
            if idx%10==0 and len(whole_tokens_list)>200000:
                print(f"Data_gathering ----> def get_uni_values()  Index: {idx}/{len(columns)} Time: {time.time() - start_time} sec.")
                start_time=time.time()
            elif idx==len(columns):
                print(f"Data_gathering ----> def get_uni_values()  Successful finished! Time: {time.time() - start_time} sec.")
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

        x=np.asarray(proart.papers_ids)
        arr_exp=np.arange(len(x))
        self.keywords_uni_corpus=proart.get_uni_values(self.all_keywords)
        self.keywords_clusters={}
        for idx, item in enumerate(self.communities_np_uni[0]):
            posi_articles=np.where(communities_np==item)[0]
            nodes_idx=nodes[posi_articles]
            
            posi_keywords_idx=proart.match_numpy(nodes_idx, x, arr_exp, None)
            print(f"Data_gathering ----> def extract_keywords(): {idx}/{len(self.communities_np_uni[0])}")
            
            keywords_idx=self.all_keywords.loc[posi_keywords_idx]
            keywords_uni_idx=self.get_uni_values(keywords_idx)
            self.keywords_clusters[idx]=keywords_uni_idx
            
        return self.keywords_clusters

    def make_xnetwork_clustered(self,network):
        print("Data_gathering ----> def make_xnetwork_clustered()")
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
        print(f"Data_gathering ----> def make_xnetwork_clustered() ----> DONE in time: {np.round(time.time()-start_time_network,3)} sec.")

    def report_dict2xlsx(self, dictionary, name,border):
        print("Data_gathering ----> def report_dict2xlsx()")
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
            print("Data_gathering ----> def report_dict2xlsx() ------> DONE")
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
    print("\t  Data_gathering ----> def match_numpy()")
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

