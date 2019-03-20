#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import zipfile


from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import downloadFromURL, removeFeatures


class BookCrossingReader(DataReader):
    """
    Collected from: http://www2.informatik.uni-freiburg.de/~cziegler/BX/

    """

    DATASET_URL = "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip"
    DATASET_SUBFOLDER = "BookCrossing/"
    AVAILABLE_ICM = ["ICM_book_crossing"]



    def __init__(self, **kwargs):
        super(BookCrossingReader, self).__init__(**kwargs)

        print("BookCrossingReader: Ratings are in range 1-10, value -1 refers to an implicit rating")
        print("BookCrossingReader: ICM contains the author, publisher, year and tokens from the title")


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        print("BookCrossingReader: Ratings are in range 1-10, value -1 refers to an implicit rating")
        print("BookCrossingReader: ICM contains the author, publisher, year and tokens from the title")


        print("BookCrossingReader: Loading original data")

        folder_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER


        try:

            dataFile = zipfile.ZipFile(folder_path + "BX-CSV-Dump.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("BookCrossingReader: Unable to find or extract data zip file. Downloading...")

            downloadFromURL(self.DATASET_URL, folder_path, "BX-CSV-Dump.zip")

            dataFile = zipfile.ZipFile(folder_path + "BX-CSV-Dump.zip")


        URM_path = dataFile.extract("BX-Book-Ratings.csv", path=folder_path + "decompressed")
        ICM_path = dataFile.extract("BX-Books.csv", path=folder_path + "decompressed")


        print("BookCrossingReader: loading ICM")
        self.ICM_book_crossing, self.tokenToFeatureMapper_ICM_book_crossing, self.item_original_ID_to_index = self._loadICM(ICM_path, separator=';', header=True, if_new_item ="add")

        self.ICM_book_crossing, _, self.tokenToFeatureMapper_ICM_book_crossing = removeFeatures(self.ICM_book_crossing, minOccurrence = 5, maxPercOccurrence = 0.30,
                                                                                                reconcile_mapper = self.tokenToFeatureMapper_ICM_book_crossing)


        #############################
        ##########
        ##########      Load metadata using AmazonReviewData
        ##########      for books ASIN corresponds to ISBN
        #
        # print("BookCrossingReader: loading ICM from AmazonReviewData")
        #
        # from Data_manager.AmazonReviewData.AmazonReviewDataReader import AmazonReviewDataReader
        #
        # # Pass "self" object as it contains the item_id mapper already initialized with the ISBN
        # self.ICM_amazon, self.tokenToFeatureMapper_ICM_amazon = AmazonReviewDataReader._loadMetadata(self, if_new_item ="add")
        #
        # self.ICM_amazon, _, self.tokenToFeatureMapper_ICM_amazon = removeFeatures(self.ICM_amazon, minOccurrence = 5, maxPercOccurrence = 0.30,
        #                                                                           reconcile_mapper=self.tokenToFeatureMapper_ICM_amazon)


        print("BookCrossingReader: loading URM")
        self.URM_all, _, self.user_original_ID_to_index = self._loadURM(URM_path, separator=";", header = True, if_new_user = "add", if_new_item = "ignore")


        print("BookCrossingReader: cleaning temporary files")

        import shutil

        shutil.rmtree(folder_path + "decompressed", ignore_errors=True)

        print("BookCrossingReader: loading complete")




    def _loadURM (self, filePath, header = True, separator="::", if_new_user = "add", if_new_item = "ignore"):



        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = self.item_original_ID_to_index, on_new_col = if_new_item,
                                                        preinitialized_row_mapper = None, on_new_row = if_new_user)


        fileHandle = open(filePath, "r", encoding='latin1')
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:

            numCells += 1
            if (numCells % 1000000 == 0):
                print("Processed {} cells".format(numCells))

            line = line.replace('"', '')

            #print(line)

            if (len(line)) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                user_id = line[0]
                item_id = line[1]

                # If 0 rating is implicit
                # To avoid removin it accidentaly, set ti to -1
                rating = float(line[2])

                if rating == 0:
                    rating = -1

                URM_builder.add_data_lists([user_id], [item_id], [rating])

        fileHandle.close()

        return  URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()




    def _loadICM(self, ICM_path, header=True, separator=',', if_new_item = "add"):

        # Pubblication Data and word in title
        from Data_manager.TagPreprocessing import tagFilterAndStemming


        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                        preinitialized_row_mapper = None, on_new_row = if_new_item)



        fileHandle = open(ICM_path, "r", encoding='latin1')
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 100000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:

                line = line.replace('"', '')
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                item_id = line[0]


                # Book Title
                featureTokenList = tagFilterAndStemming(line[1])
                # # Book author
                # featureTokenList.extend(tagFilterAndStemming(line[2]))
                # # Book year
                # featureTokenList.extend(tagFilterAndStemming(line[3]))
                # # Book publisher
                # featureTokenList.extend(tagFilterAndStemming(line[4]))

                #featureTokenList = tagFilterAndStemming(" ".join([line[1], line[2], line[3], line[4]]))

                featureTokenList.extend([line[2], line[3], line[4]])

                ICM_builder.add_single_row(item_id, featureTokenList, data=1.0)



        fileHandle.close()


        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()

