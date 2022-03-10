/*
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

package machine_learning.clusterers;

import experiments.data.DatasetLoading;
import weka.clusterers.NumberOfClustersRequestable;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import static utilities.ClusteringUtilities.createDistanceMatrix;
import static utilities.InstanceTools.deleteClassAttribute;

/**
 * Implementation of the KMedoids algorithm with
 * options for finding a value for k and a refined initial medoid selection.
 *
 * @author Matthew Middlehurst
 */
public class KMedoids extends DistanceBasedVectorClusterer implements NumberOfClustersRequestable {

    private int k = 2;
    private boolean findBestK = false;
    private boolean refinedInitialMedoids = false;
    private int numSubsamples = 10;

    private double[][] distanceMatrix;
    private int numInstances;
    private boolean hasInitialMedoids = false;
    private boolean hasDistances = false;

    private int[] medoids;

    public KMedoids() {
    }

    //Used when finding best value for k to avoid recalculating distances
    public KMedoids(double[][] distanceMatrix) {
        this.distanceMatrix = distanceMatrix;
        this.hasDistances = true;
    }

    //Used when selecting refined initial medoids.
    private KMedoids(int[] initialMedoids) {
        super();
        this.medoids = initialMedoids;
        this.hasInitialMedoids = true;
    }

    public int[] getMedoids() {
        return medoids;
    }

    @Override
    public int numberOfClusters() {
        return k;
    }

    @Override
    public void setNumClusters(int numClusters) throws Exception {
        k = numClusters;
    }

    public void setFindBestK(boolean b) {
        this.findBestK = b;
    }

    public void setRefinedInitialMedoids(boolean b) {
        this.refinedInitialMedoids = b;
    }

    public void setNumSubsamples(int n) {
        this.numSubsamples = n;
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        super.buildClusterer(data);

        numInstances = train.size();
        assignments = new double[numInstances];

        if (numInstances <= k) {
            medoids = new int[numInstances];

            for (int i = 0; i < numInstances; i++) {
                assignments[i] = i;
                medoids[i] = i;
            }

            clusters = new ArrayList[k];

            for (int i = 0; i < k; i++) {
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

        if (!hasDistances) {
            distanceMatrix = createDistanceMatrix(train, distFunc);
        }

        if (findBestK) {
            //Build clusters using multiple values of k and uses the best one.
            findBestK(train);
        } else {
            //Pick initial medoids.
            if (!hasInitialMedoids) {
                if (refinedInitialMedoids) {
                    initialMedoidsRefined(train);
                } else {
                    initialMedoids();
                }
            }

            boolean finished = false;

            //Change medoids until medoid location no longer changes.
            while (!finished) {
                calculateClusterMembership();
                finished = selectMedoids();
            }
        }

        for (int n = 0; n < numInstances; n++) {
            for (int i = 0; i < medoids.length; i++) {
                if (medoids[i] == assignments[n]) {
                    assignments[n] = i;
                    break;
                }
            }
        }
    }

    @Override
    public int clusterInstance(Instance inst) throws Exception {
        Instance newInst = copyInstances ? new DenseInstance(inst) : inst;
        int clsIdx = inst.classIndex();
        if (clsIdx >= 0){
            newInst.setDataset(null);
            newInst.deleteAttributeAt(clsIdx);
        }

        if (normaliseData)
            normaliseData(newInst);

        double minDist = Double.MAX_VALUE;
        int closestCluster = 0;
        for (int i = 0; i < medoids.length; ++i) {
            double dist = distFunc.distance(inst, train.get(medoids[i]));

            if (dist < minDist) {
                minDist = dist;
                closestCluster = i;
            }
        }

        return closestCluster;
    }

    //Returns the sum of the squared distance from each point to its cluster medoid
    public double clusterSquaredDistance() {
        double distSum = 0;

        for (int i = 0; i < k; i++) {
            for (int n = 0; n < clusters[i].size(); n++) {
                if (medoids[i] == clusters[i].get(n)) continue;

                if (medoids[i] > clusters[i].get(n)) {
                    distSum += distanceMatrix[medoids[i]][clusters[i].get(n)]
                            * distanceMatrix[medoids[i]][clusters[i].get(n)];
                } else {
                    distSum += distanceMatrix[clusters[i].get(n)][medoids[i]]
                            + distanceMatrix[clusters[i].get(n)][medoids[i]];
                }
            }
        }

        return distSum;
    }

    //Randomly select initial medoids
    private void initialMedoids() {
        medoids = new int[k];
        ArrayList<Integer> indexes = new ArrayList();

        for (int i = 0; i < numInstances; i++) {
            indexes.add(i);
        }

        Random rand;
        if (!seedClusterer) {
            rand = new Random();
        } else {
            rand = new Random(seed);
        }

        Collections.shuffle(indexes, rand);

        for (int i = 0; i < k; i++) {
            medoids[i] = indexes.get(i);
        }
    }


    //Bradley, Paul S., and Usama M. Fayyad.
    //"Refining Initial Points for K-Means Clustering."
    //ICML. Vol. 98. 1998.

    //Refined selection on initial medoids using a modified version of the
    //method above, running KMedoids over multiple subsamples then again on the
    //resulting medoids selecting the best performing one
    private void initialMedoidsRefined(Instances data) throws Exception {
        int subsampleSize = numInstances / 10;

        if (subsampleSize < k) {
            subsampleSize = k;
        }

        ArrayList<Integer> indexes = new ArrayList(numInstances);

        for (int i = 0; i < numInstances; i++) {
            indexes.add(i);
        }

        Random rand;
        if (!seedClusterer) {
            rand = new Random();
        } else {
            rand = new Random(seed);
        }

        int[][] subsampleMedoids = new int[numSubsamples][];

        //Get the resulting medoids from running KMedoids on multiple random
        //subsamples of the data
        for (int i = 0; i < numSubsamples; i++) {
            Collections.shuffle(indexes, rand);
            Instances subsample = new Instances(data, subsampleSize);

            for (int n = 0; n < subsampleSize; n++) {
                subsample.add(data.get(indexes.get(n)));
            }

            KMedoids kmedoids = new KMedoids();
            kmedoids.setDistanceFunction(distFunc);
            kmedoids.setNumClusters(k);
            kmedoids.setNormaliseData(false);
            kmedoids.setRefinedInitialMedoids(false);
            if (seedClusterer)
                kmedoids.setSeed(seed + (i + 1) * 37);
            kmedoids.buildClusterer(subsample);

            subsampleMedoids[i] = kmedoids.medoids;
        }

        //Create Instance object for subsample medoids.
        Instances medoidInsts = new Instances(data, numSubsamples * k);

        for (int i = 0; i < numSubsamples; i++) {
            for (int n = 0; n < k; n++) {
                medoidInsts.add(data.get(subsampleMedoids[i][n]));
            }
        }

        //Cluster again using subsample medoids as data and find the solution
        //with the lowest distortion using each set of medoids as the initial
        //set
        double minDist = Double.MAX_VALUE;
        int minIndex = -1;

        for (int i = 0; i < numSubsamples; i++) {
            int[] initialMedoids = new int[k];

            for (int n = 0; n < k; n++) {
                initialMedoids[n] = n + (i * k);
            }

            KMedoids kmedoids = new KMedoids(initialMedoids);
            kmedoids.setDistanceFunction(distFunc);
            kmedoids.setNumClusters(k);
            kmedoids.setNormaliseData(false);
            kmedoids.setRefinedInitialMedoids(false);
            if (seedClusterer)
                kmedoids.setSeed(seed + (i + 1) * 137);
            kmedoids.buildClusterer(medoidInsts);

            double dist = kmedoids.clusterSquaredDistance();

            if (dist < minDist) {
                minDist = dist;
                minIndex = i;
            }
        }

        medoids = subsampleMedoids[minIndex];
    }

    private void calculateClusterMembership() {
        //Set membership of each point to the closest medoid.
        for (int i = 0; i < numInstances; i++) {
            double minDist = Double.MAX_VALUE;

            for (int n = 0; n < k; n++) {
                if (medoids[n] > i) {
                    if (distanceMatrix[medoids[n]][i] < minDist) {
                        minDist = distanceMatrix[medoids[n]][i];
                        assignments[i] = medoids[n];
                    }
                }
                //If a point is a medoid set it to its own cluster.
                else if (medoids[n] == i) {
                    assignments[i] = medoids[n];
                    break;
                } else {
                    if (distanceMatrix[i][medoids[n]] < minDist) {
                        minDist = distanceMatrix[i][medoids[n]];
                        assignments[i] = medoids[n];
                    }
                }
            }
        }

        //Create and store an ArrayList for each cluster containing indexes of
        //points inside the cluster.
        clusters = new ArrayList[k];

        for (int i = 0; i < k; i++) {
            clusters[i] = new ArrayList();

            for (int n = 0; n < numInstances; n++) {
                if (medoids[i] == assignments[n]) {
                    clusters[i].add(n);
                }
            }
        }
    }

    //Select new medoids by finding the point with the lowest distance sum for
    //each cluster
    private boolean selectMedoids() {
        boolean changedMedoid = false;

        for (int i = 0; i < k; i++) {
            double minDist = Double.MAX_VALUE;
            int bestMedoid = -1;

            for (int n = 0; n < clusters[i].size(); n++) {
                double clusterDist = 0;

                for (int g = 0; g < clusters[i].size(); g++) {
                    if (clusters[i].get(n).equals(clusters[i].get(g))) continue;

                    if (clusters[i].get(n) > clusters[i].get(g)) {
                        clusterDist += distanceMatrix[clusters[i].get(n)][clusters[i].get(g)];
                    } else {
                        clusterDist += distanceMatrix[clusters[i].get(g)][clusters[i].get(n)];
                    }
                }

                if (clusterDist < minDist) {
                    minDist = clusterDist;
                    bestMedoid = clusters[i].get(n);
                }
            }

            //If a medoid changes return false to keep looping
            if (bestMedoid != medoids[i]) {
                changedMedoid = true;
                medoids[i] = bestMedoid;
            }
        }

        return !changedMedoid;
    }

    //Lletı, R., et al.
    //"Selecting variables for k-means cluster analysis by using a genetic algorithm that optimises the silhouettes."
    //Analytica Chimica Acta 515.1 (2004): 87-100.

    //Method of finding the best value for k based on the silhouette method
    //above
    private void findBestK(Instances data) throws Exception {
        int maxK = 10;
        double bestSilVal = 0;

        //For each value of K.
        for (int i = 2; i <= maxK; i++) {
            KMedoids kmedoids = new KMedoids(distanceMatrix);
            kmedoids.setDistanceFunction(distFunc);
            kmedoids.setNumClusters(i);
            kmedoids.setNormaliseData(false);
            kmedoids.setRefinedInitialMedoids(refinedInitialMedoids);
            if (seedClusterer)
                kmedoids.setSeed(seed + (i + 1) * 237);
            kmedoids.buildClusterer(data);

            double totalSilVal = 0;

            //For each cluster created by k-means.
            for (int n = 0; n < i; n++) {
                //For each point in the cluster.
                for (int g = 0; g < kmedoids.clusters[n].size(); g++) {
                    double clusterDist = 0;
                    double minOtherClusterDist = Double.MAX_VALUE;

                    int index = kmedoids.clusters[n].get(g);

                    //Find mean distance of the point to other points in its
                    //cluster.
                    for (int j = 0; j < kmedoids.clusters[n].size(); j++) {
                        if (index == kmedoids.clusters[n].get(j)) continue;

                        if (index > kmedoids.clusters[n].get(j)) {
                            clusterDist += distanceMatrix[index][kmedoids.clusters[n].get(j)];
                        } else {
                            clusterDist += distanceMatrix[kmedoids.clusters[n].get(j)][index];
                        }
                    }

                    clusterDist /= kmedoids.clusters[n].size();

                    //Find the minimum distance of the point to other clusters.
                    for (int m = 0; m < i; m++) {
                        if (m == n) {
                            continue;
                        }

                        double otherClusterDist = 0;

                        for (int j = 0; j < kmedoids.clusters[m].size(); j++) {
                            if (index > kmedoids.clusters[m].get(j)) {
                                otherClusterDist += distanceMatrix[index][kmedoids.clusters[m].get(j)];
                            } else {
                                otherClusterDist += distanceMatrix[kmedoids.clusters[m].get(j)][index];
                            }
                        }

                        otherClusterDist /= kmedoids.clusters[m].size();

                        if (otherClusterDist < minOtherClusterDist) {
                            minOtherClusterDist = otherClusterDist;
                        }
                    }

                    //Calculate the silhoutte value for the point and add it
                    //to the total
                    double silVal = minOtherClusterDist - clusterDist;
                    double div = clusterDist;

                    if (minOtherClusterDist > clusterDist) {
                        div = minOtherClusterDist;
                    }

                    silVal /= div;
                    totalSilVal += silVal;
                }
            }

            if (totalSilVal > bestSilVal) {
                bestSilVal = totalSilVal;
                medoids = kmedoids.medoids;
                assignments = kmedoids.assignments;
                clusters = kmedoids.clusters;
                k = kmedoids.k;
            }
        }
    }

    public static void main(String[] args) throws Exception {
        String[] datasets = {"Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\aggregation.arff",
                "Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\clustersynth.arff",
                "Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\dptest1k.arff",
                "Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\dptest4k.arff",
                "Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\flame.arff",
                "Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\spiral.arff"};
        String[] names = {"aggre", "synth", "dptest1k", "dptest4k", "flame", "spiral"};
        boolean output = true;

        if (output) {
            System.out.println("cd('Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\" +
                    "DensityPeakVector')");
            System.out.println("load('matlabCluster.mat')");
            System.out.println("k = [1,2,3,4,5,6,7,8,9,10]");
        }

        for (int i = 0; i < datasets.length; i++) {
            Instances inst = DatasetLoading.loadDataNullable(datasets[i]);
            inst.setClassIndex(inst.numAttributes() - 1);
            KMedoids kmedoids = new KMedoids();
            kmedoids.setFindBestK(true);
            kmedoids.setRefinedInitialMedoids(true);
            kmedoids.setSeed(0);
            kmedoids.buildClusterer(inst);

            if (output) {
                System.out.println(names[i] + "c = " + Arrays.toString(kmedoids.assignments));
                System.out.println("figure");
                System.out.println("scatter(" + names[i] + "x," + names[i] + "y,[],scatterColours(" + names[i] + "c))");
            }
        }
    }
}
