/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */

package examples;

import evaluation.storage.ClustererResults;
import experiments.data.DatasetLoading;
import machine_learning.clusterers.KMeans;
import tsml.clusterers.UnsupervisedShapelets;
import utilities.ClusteringUtilities;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Examples to show the method for building clusterers and basic usage.
 *
 * @author Matthew Middlehurst
 */
public class Clusterers {

    public static void main(String[] args) throws Exception {

        // We'll use this data throughout, see Ex01_Datahandling
        Instances inst = DatasetLoading.loadChinatown();
        System.out.println(Arrays.toString(inst.attributeToDoubleArray(inst.classIndex())));

        // Create an object from one of the time series or vector clusters implemented.
        // Call the buildClusterer method with your data. Most clusters will need the number of clusters k to be set.
        UnsupervisedShapelets us = new UnsupervisedShapelets();
        us.setSeed(0);
        us.setNumberOfClusters(inst.numClasses());
        us.buildClusterer(inst);

        // You can find the cluster assignments for each data instance by calling getAssignments() and each cluster
        // containing instance indicies using getClusters().
        // The index of assignments array will match the Instances object, i.e. index 0 with value 1 == first instance
        // of data assigned to cluster 1.
        double[] tsAssignments = us.getAssignments();
        ArrayList<Integer>[] tsClusters = us.getClusters();
        System.out.println("UnsupervisedShapelets cluster assignments:");
        System.out.println(Arrays.toString(tsAssignments));
        System.out.println("UnsupervisedShapelets clusters:");
        System.out.println(Arrays.toString(tsClusters));

        // ClustererResults is out storage for completed cluster results. The class will calculate popular clustering
        // metrics such as rand index and mutual information.
        ClustererResults tsCR = ClusteringUtilities.getClusteringResults(us, inst);
        tsCR.findAllStats();
        System.out.println("UnsupervisedShapelets results:");
        System.out.println(tsCR.statsToString());
        System.out.println();

        // Non-TSC clustering algorithms are also available in tsml.
        // weka also implements a range of clustering algorithmsm any class value must be removed prior to use these
        // however.
        KMeans km = new KMeans();
        km.setSeed(0);
        km.setNumberOfClusters(inst.numClasses());
        km.buildClusterer(inst);

        double[] vAssignments = km.getAssignments();
        ArrayList<Integer>[] vClusters = km.getClusters();
        System.out.println("KMeans cluster assignments:");
        System.out.println(Arrays.toString(vAssignments));
        System.out.println("KMeans clusters:");
        System.out.println(Arrays.toString(vClusters));

        ClustererResults vCR = ClusteringUtilities.getClusteringResults(km, inst);
        vCR.findAllStats();
        System.out.println("KMeans results:");
        System.out.println(vCR.statsToString());
    }

}
