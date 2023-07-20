import argparse
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import math

from scipy.spatial.distance import squareform


# This is the program for Homework 6 Agglomeration. The program takes in a csv file, and finds the cross-correlation
# coefficient for each attribute. It also answers some harder questions from the write-up. After that, it uses a
# method to agglomerate the data, displaying both a dendrogram, and information on the last 20 clusters
# that were merged. This process can be quite time-consuming for larger data sets, so it is recommended to use the
# -limit flag in order to save time.
#
# Author: Zachary Rutherford
# Email: zjr6302@rit.edu
# Date: March 2023

def agglomerate(data):
    """
    Use agglomeration in order to cluster the given data. First, a distance matrix is made from every value given.
    The program then outputs a dendrogram to show the process of agglomeration to the user. After that, the
    method repeatedly merges clusters, storing the last 20 smallest merged cluster sizes, including their prototypes.
    This continues until there is only 1 cluster left. This method is extremely time-consuming for large amounts of
    data. It is recommended to use a smaller amount to save time.

    :param data: the data from the csv file, minus the ids.
    """
    # setup arrays
    clusters = []
    for user in range(len(data)):
        clusters.append([user])

    # create matrix
    distance_matrix = make_distance_matrix(clusters, data)

    # print dendrogram, got code from https://www.w3schools.com/python/python_ml_hierarchial_clustering.asp
    linkage_data = linkage(squareform(distance_matrix), 'ward')
    fig, ax = plt.subplots(figsize=(15, 10))
    dendrogram(linkage_data, ax=ax, leaf_rotation=90)
    plt.title('Dendrogram')
    plt.xlabel('Users')
    plt.ylabel('Distance')
    plt.show()

    last_smallest_merged = []
    prototype_merged = []
    # iterate until only 1 cluster
    while len(clusters) > 1:
        print(len(clusters), " clusters left")
        min_distance = float('inf')
        min_cluster1 = 0
        min_cluster2 = 0
        # iterate through clusters
        for cluster1 in range(len(clusters)):
            for cluster2 in range(cluster1 + 1, len(clusters)):
                if distance_matrix[cluster1][cluster2] < min_distance:
                    min_distance = distance_matrix[cluster1][cluster2]
                    min_cluster1 = cluster1
                    min_cluster2 = cluster2

        # add to smallest merged
        if len(clusters[min_cluster1]) < len(clusters[min_cluster2]):
            last_smallest_merged.append(len(clusters[min_cluster1]))
            prototype_merged.append(make_prototype(clusters[min_cluster1], data))
        else:
            last_smallest_merged.append(len(clusters[min_cluster2]))
            prototype_merged.append(make_prototype(clusters[min_cluster2], data))

        # check if 20, and if so remove 1st
        if len(last_smallest_merged) > 20:
            last_smallest_merged.pop(0)
            prototype_merged.pop(0)

        # if the last 2 clusters print some info
        if len(clusters) == 2:
            if len(clusters[min_cluster1]) < len(clusters[min_cluster2]):
                print("length of bigger cluster:", len(clusters[min_cluster2]))
                print("prototype of bigger cluster", make_prototype(clusters[min_cluster2], data))
            else:
                print("length of bigger cluster:", len(clusters[min_cluster1]))
                print("prototype of bigger cluster", make_prototype(clusters[min_cluster1], data))

        # Merge clusters
        merged_cluster = clusters[min_cluster1] + clusters[min_cluster2]
        if min_cluster1 < min_cluster2:
            clusters.pop(min_cluster2)
            clusters.pop(min_cluster1)
        else:
            clusters.pop(min_cluster1)
            clusters.pop(min_cluster2)

        clusters.append(merged_cluster)

        # create matrix
        distance_matrix = make_distance_matrix(clusters, data)

    print("Sizes of last 20 smallest merged clusters:")
    print(last_smallest_merged)
    print("Prototypes of last 20 smallest merged clusters:")
    for protoype in prototype_merged:
        print(protoype)


def make_distance_matrix(clusters, data):
    """
    Create the distance matrix for the given clusters. Finds the distance from the prototype of each cluster
    in order to determine distance.

    :param clusters: the clusters you want make a distance matrix for.
    :param data: the dataframe containing the attributes and their values.
    :return: the distance matrix
    """
    # set up matrix
    distance_matrix = []
    for user in range(len(clusters)):
        distance_matrix.append([])

    print("Calculating Distance Matrix, this may take a while")
    # Calculate pairwise distance matrix
    for cluster1 in range(len(clusters)):
        print("\r", round(cluster1 / len(clusters) * 100, 2), "% done", end=" ")
        # only have to do half of matrix, can copy over other half
        for cluster2 in range(cluster1, len(clusters)):
            cluster1_prototype = make_prototype(clusters[cluster1], data)
            cluster2_prototype = make_prototype(clusters[cluster2], data)
            difference = 0;
            for attribute in data.columns:
                difference += math.pow(cluster1_prototype[attribute] - cluster2_prototype[attribute], 2)
            distance_matrix[cluster1].append(math.sqrt(difference))
            if cluster1 != cluster2:
                distance_matrix[cluster2].append(math.sqrt(difference))

    print("\r", round(100.00, 2), "% done", end=" ")
    print("Distance Matrix had been created")
    return distance_matrix


def make_prototype(cluster, data):
    """
    Creates a prototype from the given cluster. Uses the users that make up the cluster and finds the
    average of each attribute. This creates a prototype user at the center of the cluster.

    :param cluster: the cluster you want to find a prototype of.
    :param data: the data to create the prototype from.
    :return: the prototype
    """
    mean_attributes = {}
    for attribute in data.columns:
        attribute_sum = 0
        for user in cluster:
            attribute_sum += data.iloc[user][attribute]
        mean_attributes[attribute] = attribute_sum/len(cluster)
    return mean_attributes




def main():
    """
    The main method of the program. Parses through the arguments to get the file. Can also accept the desired number
    of values to test due to the computational complexity of the task. A limit of 100 works on my computer in about
    10 minutes. It increases exponentially, however, so I wouldn't recommend too many data points. Once the
    Cross-Correlation is computed, the program calls the Agglomerate method, which creates a dendrogram and then uses
    Agglomeration to cluster the data points in a nth degree plane, where n is the number of attributes.
    """
    # create argparse
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("path")
    parser.add_argument("-limit", required=False, type=int)
    # parse through the arguments
    args = parser.parse_args()

    # Try and open the file
    try:
        data = pd.read_csv(args.path)
    except:
        print(args.path + " does not exist")
        return

    # PART A

    # get attributes of the data, exclude the id.
    attributes = data.iloc[:, 1:]

    # get correlation matrix
    corr_matrix = attributes.corr()
    # round correlation matrix
    corr_matrix = corr_matrix.round(decimals=2)

    # print matrix, use to_string to ensure no truncation
    print(corr_matrix.to_string())

    # Find most positive
    most = -1
    most_attribute1 = ""
    most_attribute2 = ""
    # Find most negative
    least = 1
    least_attribute1 = ""
    least_attribute2 = ""
    # Find least correlated attributes
    total_correlations = {}

    # iterate through the matrix
    for row in range(len(corr_matrix.index)):
        total_attribute_correlation = 0
        for column in range(len(corr_matrix.columns)):
            total_attribute_correlation += abs(corr_matrix.iloc[row, column])
            if row != column:
                if corr_matrix.iloc[row, column] > most:
                    most = corr_matrix.iloc[row, column]
                    most_attribute1 = corr_matrix.columns[row]
                    most_attribute2 = corr_matrix.columns[column]
                if corr_matrix.iloc[row, column] < least:
                    least = corr_matrix.iloc[row, column]

                    least_attribute1 = corr_matrix.columns[row]
                    least_attribute2 = corr_matrix.columns[column]
        total_correlations[corr_matrix.columns[row]] = total_attribute_correlation

    # sort the total_correlations dictionary by value and return the two least correlated attributes
    sorted_correlations = sorted(total_correlations.items(), key=lambda x: x[1])
    least_correlated = [sorted_correlations[0][0], sorted_correlations[1][0]]

    print(f"The two attributes with the most positive cross-correlation are {most_attribute1} and {most_attribute2} at {most}.")
    print(f"The two attributes with the most negative cross-correlation are {least_attribute1} and {least_attribute2} at {least}.")
    print(f"The two least correlated attributes are {least_correlated[0]} and {least_correlated[1]}.")

    # find the largest correlation for each attribute, if it is less than 0.1 it may be irrelevant
    for row in range(len(corr_matrix.columns)):
        max_correlation = 0;
        for column in range(len(corr_matrix.columns)):
            if row != column:
                if corr_matrix.iloc[row, column] > max_correlation:
                    max_correlation = corr_matrix.iloc[row, column]
        if max_correlation < 0.1:
            print(f"{corr_matrix.columns[row]} has a maximum absolute correlation value of {max_correlation}.")

    # PART B

    # Run the test suite
    if args.limit:
        agglomerate(attributes[:args.limit])
    else:
        agglomerate(attributes)


# run the program
main()
