/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package machine_learning.clusterers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import experiments.data.DatasetLoading;
import weka.core.Instance;
import weka.core.Instances;

import static utilities.ClusteringUtilities.createDistanceMatrix;
import static utilities.InstanceTools.deleteClassAttribute;

/**
 * Implementation of the K-Means algorithm with options for finding a value
 * for k and a refined initial cluster center selection.
 *
 * @author Matthew Middlehurst
 */
public class KMeans extends AbstractVectorClusterer{

    //MacQueen, James.
    //"Some methods for classification and analysis of multivariate observations."
    //Proceedings of the fifth Berkeley symposium on mathematical statistics and probability. Vol. 1. No. 14. 1967.

    private int k = 2;
    private boolean findBestK = false;
    private boolean refinedInitialCenters = false;
    private int numSubsamples = 30;
    private int seed = Integer.MIN_VALUE;
    private int maxIterations = 200;

    private int numInstances;
    private double[][] centerDistances;
    private boolean hasInitialCenters = false;

    private Instances clusterCenters;

    public KMeans(){}

    //Used when selecting refined initial centers.
    private KMeans(Instances initialCenters){
        super();
        this.clusterCenters = new Instances(initialCenters);
        this.hasInitialCenters = true;
    }

    public Instances getClusterCenters(){
        return clusterCenters;
    }

    @Override
    public int numberOfClusters(){
        return k;
    }

    public void setNumberOfClusters(int n){ k = n; }

    public void setFindBestK(boolean b){
        this.findBestK = b;
    }

    public void setRefinedInitialMedoids(boolean b){
        this.refinedInitialCenters = b;
    }

    public void setNumSubsamples(int n){
        this.numSubsamples = n;
    }

    public void setSeed(int seed){ this.seed = seed; }

    public void setMaxIterations(int n){
        this.maxIterations = n;
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        if (copyInstances){
            data = new Instances(data);
        }

        deleteClassAttribute(data);

        distFunc.setInstances(data);
        numInstances = data.size();
        assignments = new int[numInstances];

        if (numInstances <= k){
            clusterCenters = new Instances(data);

            for (int i = 0; i < numInstances; i++){
                assignments[i] = i;
            }

            clusters = new ArrayList[k];

            for (int i = 0; i < k; i++){
                clusters[i] = new ArrayList();
            }

            for (int i = 0; i < numInstances; i++) {
                for (int n = 0; n < k; n++) {
                    if (n == assignments[i]) {
                        clusters[n].add(i);
                        break;
                    }
                }
            }

            return;
        }

        if (normaliseData){
            normaliseData(data);
        }

        if (findBestK){
            //Builds clusters using multiple values of k and keeps the best one
            findBestK(data);
        }
        else{
            //Pick initial cluster centers.
            if (refinedInitialCenters){
                initialClusterCentersRefined(data);
            }
            else if (!hasInitialCenters){
                initialClusterCenters(data);
            }

            boolean finished = false;
            int iterations = 0;

            //Change cluster centers until cluster membership no longer changes
            while(!finished){
                centerDistances = createCenterDistances(data);

                //If no clusters changed membership.
                if (!calculateClusterMembership() || iterations == maxIterations){
                    finished = true;
                }
                else{
                    selectClusterCenters(data);
                }

                iterations++;
            }
        }
    }

    //Returns the sum of the squared distance from each point to its cluster
    //center
    public double clusterSquaredDistance(Instances data){
        distFunc.setInstances(data);
        double distSum = 0;

        for (int i = 0; i < k; i++){
            for(int n = 0; n < clusters[i].size(); n++){
                double dist = distFunc.distance(clusterCenters.get(i),data.get(clusters[i].get(n)));
                distSum += dist*dist;
            }
        }

        return distSum;
    }

    //Create distances to cluster centers
    private double[][] createCenterDistances(Instances data){
        double[][] centerDists = new double[k][numInstances];

        for (int i = 0; i < numInstances; i++){
            Instance first = data.get(i);

            for (int n = 0; n < k; n++){
                centerDists[n][i] = distFunc.distance(first, clusterCenters.get(n));
            }
        }

        return centerDists;
    }

    //Randomly select initial cluster centers
    private void initialClusterCenters(Instances data){
        ArrayList<Integer> indexes = new ArrayList(numInstances);

        for (int i = 0; i < numInstances; i++){
            indexes.add(i);
        }

        Random rand;

        if (seed == Integer.MIN_VALUE){
            rand = new Random();
        } else {
            rand = new Random(seed);
        }

        clusterCenters = new Instances(data,0);
        Collections.shuffle(indexes, rand);

        for (int i = 0; i < k; i++){
            clusterCenters.add(data.get(indexes.get(i)));
        }
    }


    //Bradley, Paul S., and Usama M. Fayyad.
    //"Refining Initial Points for K-Means Clustering."
    //ICML. Vol. 98. 1998.

    //Refined selection on initial cluster centers using the method above,
    //running k-means over multiple subsamples then again on the resulting
    //centers selecting the best perfoming one
    private void initialClusterCentersRefined(Instances data) throws Exception{
        int subsampleSize = numInstances/10;

        if (subsampleSize < k){
            subsampleSize = k;
        }

        ArrayList<Integer> indexes = new ArrayList(numInstances);

        for (int i = 0; i < numInstances; i++){
            indexes.add(i);
        }

        Random rand;

        if (seed == Integer.MIN_VALUE){
            rand = new Random();
        } else {
            rand = new Random(seed);
        }

        //Randomly select centers for the subsample data
        Instances initialClusterCenters = new Instances(data,k);
        Collections.shuffle(indexes, rand);

        for (int i = 0; i < k; i++){
            initialClusterCenters.add(data.get(indexes.get(i)));
        }

        Instances[] subsampleCenters = new Instances[numSubsamples];

        //Get the resulting centers from running k-means on multiple random
        //subsamples of the data
        for (int i = 0; i < numSubsamples; i++){
            Collections.shuffle(indexes, rand);
            Instances subsample = new Instances(data, subsampleSize);

            for (int n = 0; n < subsampleSize; n++){
                subsample.add(data.get(indexes.get(n)));
            }

            boolean finished = false;

            while (!finished){
                KMeans kmeans = new KMeans(initialClusterCenters);
                kmeans.setNumberOfClusters(k);
                kmeans.setNormaliseData(false);
                kmeans.setFindBestK(false);
                kmeans.setRefinedInitialMedoids(false);
                kmeans.setSeed(seed);
                kmeans.buildClusterer(subsample);

                boolean emptyCluster = false;

                //If any cluster is empty set the initial cluster centre to the
                //point with the max distance from its centre
                for (int n = 0; n < k; n++){
                    if (kmeans.clusters[n].isEmpty()){
                        emptyCluster = true;
                        double maxDist = 0;
                        int maxIndex = -1;

                        for (int g = 0; g < subsampleSize; g++){
                            double dist = kmeans.centerDistances[kmeans.assignments[g]][g];

                            if (dist > maxDist){
                                boolean contains = false;

                                for (int j = 0; j < k; j++){
                                    if (Arrays.equals(initialClusterCenters.get(j).toDoubleArray(),subsample.get(g).toDoubleArray())){
                                        contains = true;
                                        break;
                                    }
                                }

                                if (!contains){
                                    maxDist = dist;
                                    maxIndex = g;
                                }
                            }
                        }

                        initialClusterCenters.set(n, subsample.get(maxIndex));
                    }
                }

                subsampleCenters[i] = kmeans.clusterCenters;

                if (!emptyCluster){
                    finished = true;
                }
            }
        }

        //Create Instance object for subsample centers
        Instances centers = new Instances(data,numSubsamples*k);

        for (int i = 0; i < numSubsamples; i++){
            for (int n = 0; n < k; n++){
                centers.add(subsampleCenters[i].get(n));
            }
        }

        //Cluster again using subsample centers as data and find the solution
        //with the lowest distortion using each set of centers as the initial
        //set
        double minDist = Double.MAX_VALUE;
        int minIndex = -1;

        for (int i = 0; i < numSubsamples; i++){
            KMeans kmeans = new KMeans(subsampleCenters[i]);
            kmeans.setNumberOfClusters(k);
            kmeans.setNormaliseData(false);
            kmeans.setFindBestK(false);
            kmeans.setRefinedInitialMedoids(false);
            kmeans.setSeed(seed);
            kmeans.buildClusterer(centers);

            double dist = kmeans.clusterSquaredDistance(centers);

            if (dist < minDist){
                minDist = dist;
                minIndex = i;
            }
        }

        clusterCenters = subsampleCenters[minIndex];
    }

    private boolean calculateClusterMembership(){
        boolean membershipChange = false;

        //Set membership of each point to the closest cluster center
        for (int i = 0; i < numInstances; i++){
            double minDist = Double.MAX_VALUE;
            int minIndex = -1;

            for (int n = 0; n < k; n++){
                if (centerDistances[n][i] < minDist){
                    minDist = centerDistances[n][i];
                    minIndex = n;
                }
            }

            //If membership of any point changed return true to keep
            //looping
            if (minIndex != assignments[i]){
                assignments[i] = minIndex;
                membershipChange = true;
            }
        }

        //Create and store an ArrayList for each cluster containing indexes of
        //points inside the cluster
        clusters = new ArrayList[k];

        for (int i = 0; i < k; i++){
            clusters[i] = new ArrayList();
        }

        for (int i = 0; i < numInstances; i++){
            for (int n = 0; n < k; n++){
                if(n == assignments[i]){
                    clusters[n].add(i);
                    break;
                }
            }
        }

        return membershipChange;
    }

    //Select the new cluster centers for each cluster to be the mean of each
    //attribute of points in the cluster.
    private void selectClusterCenters(Instances data){
        for (int i = 0; i < k; i++){
            Instance center = clusterCenters.get(i);

            for (int n = 0; n < data.numAttributes()-1; n++){
                double sum = 0;

                for (Integer g : clusters[i]){
                    sum += data.get(g).value(n);
                }

                center.setValue(n, sum/clusters[i].size());
            }
        }
    }

    //Lletı, R., et al.
    //"Selecting variables for k-means cluster analysis by using a genetic algorithm that optimises the silhouettes."
    //Analytica Chimica Acta 515.1 (2004): 87-100.

    //Method of finding the best value for k based on the silhouette method
    //above
    private void findBestK(Instances data) throws Exception{
        int maxK = 10;
        double bestSilVal = 0;

        double[][] distMatrix = createDistanceMatrix(data, distFunc);

        //For each value of K
        for (int i = 2; i <= maxK; i++){
            KMeans kmeans = new KMeans();
            kmeans.setNumberOfClusters(i);
            kmeans.setNormaliseData(false);
            kmeans.setFindBestK(false);
            kmeans.setRefinedInitialMedoids(refinedInitialCenters);
            kmeans.setSeed(seed);
            kmeans.buildClusterer(data);

            double totalSilVal = 0;

            //For each cluster created by k-means
            for (int n = 0; n < i; n++){
                //For each point in the cluster
                for (int g = 0; g < kmeans.clusters[n].size(); g++){
                    double clusterDist = 0;
                    double minOtherClusterDist = Double.MAX_VALUE;

                    int index = kmeans.clusters[n].get(g);

                    //Find mean distance of the point to other points in its
                    //cluster
                    for (int j = 0; j < kmeans.clusters[n].size(); j++){
                        if (index > kmeans.clusters[n].get(j)){
                            clusterDist += distMatrix[index][kmeans.clusters[n].get(j)];
                        }
                        else {
                            clusterDist += distMatrix[kmeans.clusters[n].get(j)][index];
                        }
                    }

                    clusterDist /= kmeans.clusters[n].size();

                    //Find the minimum distance of the point to other clusters
                    for (int m = 0; m < i; m++){
                        if(m == n){
                            continue;
                        }

                        double otherClusterDist = 0;

                        for (int j = 0; j < kmeans.clusters[m].size(); j++){
                            if (index > kmeans.clusters[m].get(j)){
                                otherClusterDist += distMatrix[index][kmeans.clusters[m].get(j)];
                            }
                            else {
                                otherClusterDist += distMatrix[kmeans.clusters[m].get(j)][index];
                            }
                        }

                        otherClusterDist /= kmeans.clusters[m].size();

                        if(otherClusterDist < minOtherClusterDist){
                            minOtherClusterDist = otherClusterDist;
                        }
                    }

                    //Calculate the silhoutte value for the point and add it
                    //to the total.
                    double silVal = minOtherClusterDist - clusterDist;
                    double div = clusterDist;

                    if(minOtherClusterDist > clusterDist){
                        div = minOtherClusterDist;
                    }

                    silVal /= div;
                    totalSilVal += silVal;
                }
            }

            if (totalSilVal > bestSilVal){
                bestSilVal = totalSilVal;
                clusterCenters = kmeans.clusterCenters;
                assignments = kmeans.assignments;
                clusters = kmeans.clusters;
                k = kmeans.k;
            }
        }
    }

    public static void main(String[] args) throws Exception{
        String[] datasets = {"Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\aggregation.arff",
                "Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\clustersynth.arff",
                "Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\dptest1k.arff",
                "Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\dptest4k.arff",
                "Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\flame.arff",
                "Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\spiral.arff"};
        String[] names = {"aggre", "synth", "dptest1k", "dptest4k", "flame", "spiral"};
        boolean output = true;

        if (output){
            System.out.println("cd('Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\" +
                    "DensityPeakVector')");
            System.out.println("load('matlabCluster.mat')");
            System.out.println("k = [1,2,3,4,5,6,7,8,9,10]");
        }

        for (int i = 0; i < datasets.length; i++){
            Instances inst = DatasetLoading.loadDataNullable(datasets[i]);
            inst.setClassIndex(inst.numAttributes()-1);
            KMeans kmeans = new KMeans();
            kmeans.setFindBestK(true);
            kmeans.setRefinedInitialMedoids(true);
            kmeans.setSeed(1);
            kmeans.buildClusterer(inst);

            if(output){
                System.out.println(names[i] + "c = " + Arrays.toString(kmeans.assignments));
                System.out.println("figure");
                System.out.println("scatter(" + names[i] + "x," + names[i] + "y,[],scatterColours(" + names[i] + "c))");
            }
        }
    }
}
