from core.cs_data_gathering import OpenAlexDataGatherer
#from core.cs_data_processing import NetworkBuilder
#from core.cs_analyzing import Analyzer
#from core.cs_plotting import Plotter


def main():
    # 1. Gather data
    gatherer = OpenAlexDataGatherer(path="./data/", folder="",email="jaromir.klarak@savba.sk")
    df_articles = gatherer.make_db_articles(    terms="machine learning", filter_by="abstract", max_per_term=10   )             # how many articles to retrieve per author

    # 2. Process data
    # load data frames
    # builder = NetworkBuilder(articles_df)
    # G = builder.build_citation_network(citations_df)

    # 3. Analyze
    # analyzer = Analyzer()
    # sim = analyzer.compute_cosine_similarity(feature_matrix)

    # 4. Plot
    # plotter = Plotter()
    # plotter.plot_network(G)

if __name__ == "__main__":
    print("Start")    
    main()    
    
    



