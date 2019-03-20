#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/01/18

@author: Maurizio Ferrari Dacrema
"""


import ast, gzip


from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import downloadFromURL, load_CSV_into_SparseBuilder, removeFeatures



def parse_json(file_path):
    g = open(file_path, 'r')

    for l in g:
        try:
            yield ast.literal_eval(l)
        except Exception as exception:
            print("Exception: {}. Skipping".format(str(exception)))




class AmazonReviewDataReader(DataReader):

    DATASET_SUBFOLDER = "AmazonReviewData/"



    def _get_ICM_metadata_path(self, data_folder, compressed_file_name, decompressed_file_name, file_url):
        """
        Metadata files are .csv
        :param data_folder:
        :param file_name:
        :param file_url:
        :return:
        """


        try:

            open(data_folder + decompressed_file_name, "r")

        except FileNotFoundError:

            print("AmazonReviewDataReader: Decompressing metadata file...")

            try:

                decompressed_file = open(data_folder + decompressed_file_name, "wb")

                compressed_file = gzip.open(data_folder + compressed_file_name, "rb")
                decompressed_file.write(compressed_file.read())

                compressed_file.close()
                decompressed_file.close()

            except (FileNotFoundError, Exception):

                print("AmazonReviewDataReader: Unable to find or decompress compressed file. Downloading...")

                downloadFromURL(file_url, data_folder, compressed_file_name)

                decompressed_file = open(data_folder + decompressed_file_name, "wb")

                compressed_file = gzip.open(data_folder + compressed_file_name, "rb")
                decompressed_file.write(compressed_file.read())

                compressed_file.close()
                decompressed_file.close()


        return data_folder + decompressed_file_name






    def _get_URM_review_path(self, data_folder, file_name, file_url):
        """
        Metadata files are .csv
        :param data_folder:
        :param file_name:
        :param file_url:
        :return:
        """


        try:

            open(data_folder + file_name, "r")

        except FileNotFoundError:

            print("AmazonReviewDataReader: Unable to find or open review file. Downloading...")

            downloadFromURL(file_url, data_folder, file_name)


        return data_folder + file_name


    #
    # def _load_preprocessed_data_all_amazon_datasets(self, splitSubfolder = DataReader.DATASET_SUBFOLDER_ORIGINAL, ICM_to_load = None):
    #     """
    #     Loads ICM and URM from "original" subfolder
    #     :param splitSubfolder:
    #     :param ICM_to_load:
    #     :return:
    #     """
    #
    #
    #     if ICM_to_load not in self.AVAILABLE_ICM and ICM_to_load is not None:
    #         raise ValueError("DataReader: ICM to load not recognized. Available values are {}, passed was '{}'".format(self.AVAILABLE_ICM, ICM_to_load))
    #
    #     self.ICM_to_load = ICM_to_load
    #
    #
    #     print("DataReader: loading data...")
    #
    #     self.data_path = "./data/" + self.DATASET_SUBFOLDER + splitSubfolder
    #
    #     try:
    #         if splitSubfolder == DataReader.DATASET_SUBFOLDER_ORIGINAL:
    #             self.URM_all = sps.load_npz(self.data_path + "URM_all.npz")
    #
    #         else:
    #             for URM_name in self.AVAILABLE_URM:
    #                 setattr(self, URM_name, sps.load_npz(self.data_path + "{}.npz".format(URM_name)))
    #
    #
    #
    #         if ICM_to_load is None:
    #
    #             for ICM_name in self.AVAILABLE_ICM:
    #
    #                 icm_split_list = []
    #
    #                 for icm_split_index in range(self.NUM_ICM_SPLIT):
    #                     icm_split_object = sps.load_npz(self.data_path + "{}_{}.npz".format(ICM_name, icm_split_index))
    #                     icm_split_list.append(icm_split_object)
    #
    #                 setattr(self, ICM_name, sps.hstack(icm_split_list))
    #
    #
    #         else:
    #             setattr(self, ICM_to_load, sps.load_npz(self.data_path + "{}.npz".format(ICM_to_load)))
    #
    #
    #         self.load_mappers()
    #
    #         print("DataReader: loading complete")
    #
    #
    #     except FileNotFoundError as splitNotFoundException:
    #
    #         print("DataReader: URM or ICM not found")
    #
    #         raise splitNotFoundException
    #



    def _load_from_original_file_all_amazon_datasets(self, URM_path, metadata_path = None, reviews_path = None):
        # Load data from original

        print("AmazonReviewDataReader: Loading original data")


        print("AmazonReviewDataReader: loading URM")
        self.URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, separator=",", header = False)


        if metadata_path is not None:
            print("AmazonReviewDataReader: loading metadata")
            self.ICM_metadata, self.tokenToFeatureMapper_ICM_metadata, _ = self._loadMetadata(metadata_path, if_new_item ="ignore")

            self.ICM_metadata, _, self.tokenToFeatureMapper_ICM_metadata = removeFeatures(self.ICM_metadata, minOccurrence = 5, maxPercOccurrence = 0.30,
                                                                                          reconcile_mapper=self.tokenToFeatureMapper_ICM_metadata)


        if reviews_path is not None:
            print("AmazonReviewDataReader: loading reviews")
            self.ICM_reviews, self.tokenToFeatureMapper_ICM_reviews, _ = self._loadReviews(reviews_path, if_new_item ="ignore")

            self.ICM_reviews, _, self.tokenToFeatureMapper_ICM_reviews = removeFeatures(self.ICM_reviews, minOccurrence = 5, maxPercOccurrence = 0.30,
                                                                                        reconcile_mapper=self.tokenToFeatureMapper_ICM_reviews)


        # Clean temp files
        print("AmazonReviewDataReader: cleaning temporary files")

        import os

        if metadata_path is not None:
            os.remove(metadata_path)

        if reviews_path is not None:
            os.remove(reviews_path)

        print("AmazonBooksReader: loading complete")







    def _loadMetadata(self, file_path, if_new_item = "ignore"):


        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                        preinitialized_row_mapper = self.item_original_ID_to_index, on_new_row = if_new_item)


        from Data_manager.TagPreprocessing import tagFilterAndStemming, tagFilter
        import itertools


        parser_metadata = parse_json(file_path)

        numMetadataParsed = 0

        for newMetadata in parser_metadata:

            numMetadataParsed+=1
            if (numMetadataParsed % 20000 == 0):
                print("Processed {}".format(numMetadataParsed))

            item_ID = newMetadata["asin"]

            # The file might contain other elements, restrict to
            # Those in the URM

            tokenList = []

            #item_price = newMetadata["price"]

            if "title" in newMetadata:
                item_name = newMetadata["title"]
                tokenList.append(item_name)

            # Sometimes brand is not present
            if "brand" in newMetadata:
                item_brand = newMetadata["brand"]
                tokenList.append(item_brand)

            # Categories are a list of lists. Unclear whether only the first element contains data or not
            if "categories" in newMetadata:
                item_categories = newMetadata["categories"]
                item_categories = list(itertools.chain.from_iterable(item_categories))
                tokenList.extend(item_categories)


            if "description" in newMetadata:
                item_description = newMetadata["description"]
                tokenList.append(item_description)


            tokenList = ' '.join(tokenList)

            # Remove non alphabetical character and split on spaces
            tokenList = tagFilterAndStemming(tokenList)

            # Remove duplicates
            tokenList = list(set(tokenList))

            ICM_builder.add_single_row(item_ID, tokenList, data=1.0)


        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()







    def _loadReviews(self, file_path, if_new_item = "add"):


        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                        preinitialized_row_mapper = self.item_original_ID_to_index, on_new_row = if_new_item)




        from Data_manager.TagPreprocessing import tagFilterAndStemming, tagFilter


        parser_reviews = parse_json(file_path)

        numReviewParsed = 0

        for newReview in parser_reviews:

            numReviewParsed+=1
            if (numReviewParsed % 20000 == 0):
                print("Processed {} reviews".format(numReviewParsed))

            user_ID = newReview["reviewerID"]
            item_ID = newReview["asin"]

            reviewText = newReview["reviewText"]
            reviewSummary = newReview["summary"]

            tagList = ' '.join([reviewText, reviewSummary])

            # Remove non alphabetical character and split on spaces
            tagList = tagFilterAndStemming(tagList)

            ICM_builder.add_single_row(item_ID, tagList, data=1.0)




        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()

