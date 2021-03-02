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

import experiments.data.DatasetLoading;
import tsml.clusterers.UnsupervisedShapelets;
import utilities.ClusteringUtilities;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

import java.util.Arrays;

import static utilities.InstanceTools.deleteClassAttribute;

/**
 * Examples to show the method for building clusterers and basic usage.
 * 
 * @author Matthew Middlehurst (m.middlehurst@uea.ac.uk)
 */
public class Clusterers {

    public static void main(String[] args) throws Exception {
        
        // We'll use this data throughout, see Ex01_Datahandling
        int seed = 0;
        Instances[] trainTest = DatasetLoading.sampleItalyPowerDemand(seed);
        Instances inst = trainTest[0];
        Instances inst2 = trainTest[1];
        inst.addAll(inst2);

        // Create an object from one of the time series or vector clusters implemented.
        // Call the buildClusterer method with your data. Most clusters will need the number of clusters k to be set.
        UnsupervisedShapelets us = new UnsupervisedShapelets();
        us.setNumberOfClusters(inst.numClasses());
        us.buildClusterer(inst);

        // You can find the cluster assignments for each data instance by calling getAssignments().
        // The index of assignments array will match the Instances object, i.e. index 0 with value 1 == first instance
        // of data assigned to cluster 1.
        int[] tsAssignments = us.getAssignments();
        System.out.println("UnsupervisedShapelets cluster assignments:");
        System.out.println(Arrays.toString(tsAssignments));

        // A popular metric for cluster evaluation is the Rand index. A utility method is available for calculating
        // this.
        double tsRandIndex = ClusteringUtilities.randIndex(tsAssignments, inst);
        System.out.println("UnsupervisedShapelets Rand index:");
        System.out.println(tsRandIndex);

        // weka also implements a range of clustering algorithms. Any class value must be removed prior to use.
        Instances copy = new Instances(inst);
        deleteClassAttribute(copy);
        SimpleKMeans km = new SimpleKMeans();
        km.setNumClusters(inst.numClasses());
        km.setPreserveInstancesOrder(true);
        km.buildClusterer(copy);

        int[] wekaAssignments = km.getAssignments();
        System.out.println("SimpleKMeans cluster assignments:");
        System.out.println(Arrays.toString(wekaAssignments));

        double wekaRandIndex = ClusteringUtilities.randIndex(wekaAssignments, inst);
        System.out.println("SimpleKMeans Rand index:");
        System.out.println(wekaRandIndex);
    }
    
}
