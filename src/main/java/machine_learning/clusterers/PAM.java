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
import weka.core.Instances;

import static utilities.ClusteringUtilities.createDistanceMatrix;
import static utilities.InstanceTools.deleteClassAttribute;

/**
 * Implementation of the Partitioning Around Medoids (PAM) algorithm with
 * options for finding a value for k and a refined initial medoid selection.
 *
 * @author Matthew Middlehurst
 */
public class PAM extends AbstractVectorClusterer{

    //L. Kaufman, P.J. Rousseeuw
    //Finding groups in data: An introduction to cluster analysis
    //Wiley, New York (1990)

    private int k = 2;
    private boolean findBestK = false;
    private boolean refinedInitialMedoids = false;
    private int numSubsamples = 30;
    private int seed = Integer.MIN_VALUE;

    private double[][] distanceMatrix;
    private int numInstances;
    boolean hasInitialMedoids = false;
    private boolean hasDistances = false;

    private int[] medoids;

    public PAM(){}

    //Used when finding best value for k to avoid recalculating distances
    public PAM(double[][] distanceMatrix){
        this.distanceMatrix = distanceMatrix;
        this.hasDistances = true;
    }

    //Used when selecting refined initial medoids.
    private PAM(int[] initialMedoids){
        super();
        this.medoids = initialMedoids;
        this.hasInitialMedoids = true;
    }

    public int[] getMedoids(){
        return medoids;
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
        this.refinedInitialMedoids = b;
    }

    public void setNumSubsamples(int n){
        this.numSubsamples = n;
    }

    public void setSeed(int seed){
        this.seed = seed;
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        if (copyInstances){
            data = new Instances(data);
        }

        deleteClassAttribute(data);

        numInstances = data.size();
        assignments = new int[numInstances];

        if (numInstances <= k){
            medoids = new int[numInstances];

            for (int i = 0; i < numInstances; i++){
                assignments[i] = i;
                medoids[i] = i;
            }

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

            return;
        }

        if (normaliseData){
            normaliseData(data);
        }


        if (!hasDistances){
            distanceMatrix = createDistanceMatrix(data, distFunc);
        }

        if (findBestK){
            //Build clusters using multiple values of k and uses the best one.
            findBestK(data);
        }
        else{
            //Pick initial medoids.
            if (refinedInitialMedoids){
                initialMedoidsRefined(data);
            }
            else {
                initialMedoids();
            }

            boolean finished = false;

            //Change medoids until medoid location no longer changes.
            while(!finished){
                calculateClusterMembership();
                finished = selectMedoids();
            }
        }

        for (int n = 0; n < numInstances; n++){
            for (int i = 0; i < medoids.length; i++){
                if(medoids[i] == assignments[n]){
                    assignments[n] = i;
                    break;
                }
            }
        }
    }

    //Returns the sum of the squared distance from each point to its cluster
    //medoid
    public double clusterSquaredDistance(){
        double distSum = 0;

        for (int i = 0; i < k; i++){
            for(int n = 0; n < clusters[i].size(); n++){
                if (medoids[i] == clusters[i].get(n)) continue;

                if (medoids[i] > clusters[i].get(n)){
                    distSum += distanceMatrix[medoids[i]][clusters[i].get(n)]
                        * distanceMatrix[medoids[i]][clusters[i].get(n)];
                }
                else {
                    distSum += distanceMatrix[clusters[i].get(n)][medoids[i]]
                            + distanceMatrix[clusters[i].get(n)][medoids[i]];
                }
            }
        }

        return distSum;
    }

    //Randomly select initial medoids
    private void initialMedoids(){
        medoids = new int[k];
        ArrayList<Integer> indexes = new ArrayList();

        for (int i = 0; i < numInstances; i++){
            indexes.add(i);
        }

        Random rand;

        if (seed == Integer.MIN_VALUE){
            rand = new Random();
        } else {
            rand = new Random(seed);
        }

        Collections.shuffle(indexes, rand);

        for (int i = 0; i < k; i++){
            medoids[i] = indexes.get(i);
        }
    }


    //Bradley, Paul S., and Usama M. Fayyad.
    //"Refining Initial Points for K-Means Clustering."
    //ICML. Vol. 98. 1998.

    //Refined selection on initial medoids using a modified version of the
    //method above, running PAM over multiple subsamples then again on the
    //resulting medoids selecting the best perfoming one
    private void initialMedoidsRefined(Instances data) throws Exception{
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

        int[][] subsampleMedoids = new int[numSubsamples][];

        //Get the resulting medoids from running PAM on multiple random
        //subsamples of the data
        for (int i = 0; i < numSubsamples; i++){
            Collections.shuffle(indexes, rand);
            Instances subsample = new Instances(data, subsampleSize);

            for (int n = 0; n < subsampleSize; n++){
                subsample.add(data.get(indexes.get(n)));
            }

            PAM pam = new PAM();
            pam.setNumberOfClusters(k);
            pam.setNormaliseData(false);
            pam.setRefinedInitialMedoids(false);
            pam.setSeed(seed);
            pam.buildClusterer(subsample);

            subsampleMedoids[i] = pam.medoids;
        }

        //Create Instance object for subsample medoids.
        Instances medoidInsts = new Instances(data,numSubsamples*k);

        for (int i = 0; i < numSubsamples; i++){
            for (int n = 0; n < k; n++){
                medoidInsts.add(data.get(subsampleMedoids[i][n]));
            }
        }

        //Cluster again using subsample medoids as data and find the solution
        //with the lowest distortion using each set of medoids as the initial
        //set
        double minDist = Double.MAX_VALUE;
        int minIndex = -1;

        for (int i = 0; i < numSubsamples; i++){
            int[] initialMedoids = new int[k];

            for (int n = 0; n < k; n++){
                initialMedoids[n] = n + (i*k);
            }

            PAM pam = new PAM(initialMedoids);
            pam.setNumberOfClusters(k);
            pam.setNormaliseData(false);
            pam.setRefinedInitialMedoids(false);
            pam.setSeed(seed+i);
            pam.buildClusterer(medoidInsts);

            double dist = pam.clusterSquaredDistance();

            if (dist < minDist){
                minDist = dist;
                minIndex = i;
            }
        }

        medoids = subsampleMedoids[minIndex];
    }

    private void calculateClusterMembership(){
        //Set membership of each point to the closest medoid.
        for (int i = 0; i < numInstances; i++){
            double minDist = Double.MAX_VALUE;

            for (int n = 0; n < k; n++){
                if (medoids[n] > i){
                    if (distanceMatrix[medoids[n]][i] < minDist){
                        minDist = distanceMatrix[medoids[n]][i];
                        assignments[i] = medoids[n];
                    }
                }
                //If a point is a medoid set it to its own cluster.
                else if (medoids[n] == i){
                    assignments[i] = medoids[n];
                    break;
                }
                else {
                    if (distanceMatrix[i][medoids[n]] < minDist){
                        minDist = distanceMatrix[i][medoids[n]];
                        assignments[i] = medoids[n];
                    }
                }
            }
        }

        //Create and store an ArrayList for each cluster containing indexes of
        //points inside the cluster.
        clusters = new ArrayList[k];

        for (int i = 0; i < k; i++){
            clusters[i] = new ArrayList();

            for (int n = 0; n < numInstances; n++){
                if(medoids[i] == assignments[n]){
                    clusters[i].add(n);
                }
            }
        }
    }

    //Select new medoids by finding the point with the lowest distance sum for
    //each cluster
    private boolean selectMedoids(){
        boolean changedMedoid = false;

        for (int i = 0; i < k; i++){
            double minDist = Double.MAX_VALUE;
            int bestMedoid = -1;

            for (int n = 0; n < clusters[i].size(); n++){
                double clusterDist = 0;

                for (int g = 0; g < clusters[i].size(); g++){
                    if (clusters[i].get(n) == clusters[i].get(g)) continue;

                    if (clusters[i].get(n) > clusters[i].get(g)){
                        clusterDist += distanceMatrix[clusters[i].get(n)][clusters[i].get(g)];
                    }
                    else {
                        clusterDist += distanceMatrix[clusters[i].get(g)][clusters[i].get(n)];
                    }
                }

                if (clusterDist < minDist){
                    minDist = clusterDist;
                    bestMedoid = clusters[i].get(n);
                }
            }

            //If a medoid changes return false to keep looping
            if (bestMedoid != medoids[i]){
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
    private void findBestK(Instances data) throws Exception{
        int maxK = 10;
        double bestSilVal = 0;

        //For each value of K.
        for (int i = 2; i <= maxK; i++){
            PAM pam = new PAM(distanceMatrix);
            pam.setNumberOfClusters(i);
            pam.setNormaliseData(false);
            pam.setRefinedInitialMedoids(refinedInitialMedoids);
            pam.setSeed(seed);
            pam.buildClusterer(data);

            double totalSilVal = 0;

            //For each cluster created by k-means.
            for (int n = 0; n < i; n++){
                //For each point in the cluster.
                for (int g = 0; g < pam.clusters[n].size(); g++){
                    double clusterDist = 0;
                    double minOtherClusterDist = Double.MAX_VALUE;

                    int index = pam.clusters[n].get(g);

                    //Find mean distance of the point to other points in its
                    //cluster.
                    for (int j = 0; j < pam.clusters[n].size(); j++){
                        if (index == pam.clusters[n].get(j)) continue;

                        if (index > pam.clusters[n].get(j)){
                            clusterDist += distanceMatrix[index][pam.clusters[n].get(j)];
                        }
                        else {
                            clusterDist += distanceMatrix[pam.clusters[n].get(j)][index];
                        }
                    }

                    clusterDist /= pam.clusters[n].size();

                    //Find the minimum distance of the point to other clusters.
                    for (int m = 0; m < i; m++){
                        if(m == n){
                            continue;
                        }

                        double otherClusterDist = 0;

                        for (int j = 0; j < pam.clusters[m].size(); j++){
                            if (index > pam.clusters[m].get(j)){
                                otherClusterDist += distanceMatrix[index][pam.clusters[m].get(j)];
                            }
                            else {
                                otherClusterDist += distanceMatrix[pam.clusters[m].get(j)][index];
                            }
                        }

                        otherClusterDist /= pam.clusters[m].size();

                        if(otherClusterDist < minOtherClusterDist){
                            minOtherClusterDist = otherClusterDist;
                        }
                    }

                    //Calculate the silhoutte value for the point and add it
                    //to the total
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
                medoids = pam.medoids;
                assignments = pam.assignments;
                clusters = pam.clusters;
                k = pam.k;
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
            PAM pam = new PAM();
            pam.setFindBestK(true);
            pam.setRefinedInitialMedoids(true);
            pam.setSeed(1);
            pam.buildClusterer(inst);

            if(output){
                System.out.println(names[i] + "c = " + Arrays.toString(pam.assignments));
                System.out.println("figure");
                System.out.println("scatter(" + names[i] + "x," + names[i] + "y,[],scatterColours(" + names[i] + "c))");
            }
        }
    }
}
