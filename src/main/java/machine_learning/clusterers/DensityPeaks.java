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

import experiments.data.DatasetLoading;
import utilities.GenericTools;
import weka.core.Instances;

import static utilities.ClusteringUtilities.createDistanceMatrix;
import static utilities.InstanceTools.deleteClassAttribute;

/**
 * Implementation of the Density Peaks algorithm described in "Clustering by
 * fast search and find of density peaks.".
 *
 * @author Matthew Middlehurst
 */
public class DensityPeaks extends AbstractVectorClusterer {

    //Rodriguez, Alex, and Alessandro Laio.
    //"Clustering by fast search and find of density peaks."
    //Science 344.6191 (2014): 1492-1496.

    private boolean gaussianKernel = true;
    private boolean cutoffOutlierSelection = false;
    private boolean haloOutlierSelection = false;
    private double distC = -1;
    private double clusterCenterCutoff = -1;
    private double outlierCutoff = -1;

    private double[][] distanceMatrix;
    private double[] localDensities;
    private double[] shortestDist;
    private int[] nearestNeighbours;
    private int numInstances;
    private Integer[] sortedDensitiesIndex;

    private ArrayList<Integer> clusterCenters;

    public DensityPeaks(){}

    public ArrayList<Integer> getClusterCenters(){
        return clusterCenters;
    }

    @Override
    public int numberOfClusters(){
        return clusterCenters.size();
    }

    public void setGaussianKernel(boolean b){
        this.gaussianKernel = b;
    }

    public void setCutoffOutlierSelection(boolean b){
        this.cutoffOutlierSelection = b;
    }

    public void setHaloOutlierSelection(boolean b){
        this.haloOutlierSelection = b;
    }

    public void setDistC(double distC){
        this.distC = distC;
    }

    public void setClusterCenterCutoff(double cutoff){
        this.clusterCenterCutoff = cutoff;
    }

    public void setOutlierCutoff(double cutoff){
        this.outlierCutoff = cutoff;
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        if (copyInstances){
            data = new Instances(data);
        }

        deleteClassAttribute(data);

        if (normaliseData){
            normaliseData(data);
        }

        numInstances = data.size();
        distanceMatrix = createDistanceMatrix(data, distFunc);

        if (distC < 0){
            distC = getDistCDefault();
        }

        if (gaussianKernel){
            generateLocalDensitiesGuassian();
        }
        else{
            generateLocalDensitiesCutoff();
        }

        generateHighDensDistance();
        findClusterCentres();
        assignClusters();

        //Create and store an ArrayList for each cluster containing indexes of
        //points inside the cluster
        clusters = new ArrayList[clusterCenters.size()];

        for (int i = 0; i < clusterCenters.size(); i++){
            clusters[i] = new ArrayList();

            for (int n = 0; n < numInstances; n++){
                if(clusterCenters.get(i) == assignments[n]){
                    clusters[i].add(n);
                    assignments[n] = i;
                }
            }
        }
    }

    //Method used in the original implementation to set distC so that the
    //average number of neighbors is around 1 to 2% of the total number of
    //points in the dataset
    private double getDistCDefault(){
        ArrayList<Double> sortedDistances = new ArrayList<>(numInstances);

        for (int i = 0; i < numInstances; i++){
            for (int n = 0; n < i; n++){
                sortedDistances.add(distanceMatrix[i][n]);
            }
        }

        Collections.sort(sortedDistances);

        double percent = 2.0;
        int position = (int)(sortedDistances.size() * percent / 100);
        return sortedDistances.get(position-1);
    }

    //Gets the local density for each instance i with the density defined as the
    //number of points closer than distC to i
    private void generateLocalDensitiesCutoff(){
        localDensities = new double[numInstances];

        for (int i = 0; i < numInstances; i++){
            for (int n = 0; n < i; n++){
                if (distanceMatrix[i][n] - distC < 0){
                    localDensities[i]++;
                    localDensities[n]++;
                }
            }
        }
    }

    //Gets the local density for each instance i with the density estimated
    //using a gaussian kernel
    private void generateLocalDensitiesGuassian(){
        localDensities = new double[numInstances];

        for (int i = 0; i < numInstances; i++){
            for (int n = 0; n < i; n++){
                double j = distanceMatrix[i][n] / distC;
                j = Math.exp(-(j*j));

                localDensities[i] += j;
                localDensities[n] += j;
            }
        }
    }

    private void generateHighDensDistance(){
        //Find the indexes of the local densities sorted in descending order
        sortedDensitiesIndex = new Integer[numInstances];
        for (int i = 0; i < numInstances; i++){
            sortedDensitiesIndex[i] = i;
        }

        GenericTools.SortIndexDescending sort = new GenericTools.SortIndexDescending(localDensities);
        Arrays.sort(sortedDensitiesIndex, sort);

        shortestDist = new double[numInstances];
        nearestNeighbours = new int[numInstances];

        //Find the shortest distance/nearest neigbour from points with a higher
        //local density for each point
        for (int i = 1; i < numInstances; i++){
            shortestDist[sortedDensitiesIndex[i]] = Double.MAX_VALUE;

            for (int n = 0; n < i; n++){
                if (sortedDensitiesIndex[n] > sortedDensitiesIndex[i]){
                    if (distanceMatrix[sortedDensitiesIndex[n]][sortedDensitiesIndex[i]] < shortestDist[sortedDensitiesIndex[i]]){
                        shortestDist[sortedDensitiesIndex[i]] = distanceMatrix[sortedDensitiesIndex[n]][sortedDensitiesIndex[i]];
                        nearestNeighbours[sortedDensitiesIndex[i]] = sortedDensitiesIndex[n];
                    }
                }
                else {
                    if (distanceMatrix[sortedDensitiesIndex[i]][sortedDensitiesIndex[n]] < shortestDist[sortedDensitiesIndex[i]]){
                        shortestDist[sortedDensitiesIndex[i]] = distanceMatrix[sortedDensitiesIndex[i]][sortedDensitiesIndex[n]];
                        nearestNeighbours[sortedDensitiesIndex[i]] = sortedDensitiesIndex[n];
                    }
                }
            }
        }

        //Set the shortest distance of the point with the highest local density
        //to the max of the distances from other points
        double maxDensDist = 0;
        for (int i = 0; i < shortestDist.length; i++){
            if (shortestDist[i] > maxDensDist){
                maxDensDist = shortestDist[i];
            }
        }

        shortestDist[sortedDensitiesIndex[0]] = maxDensDist;
        nearestNeighbours[sortedDensitiesIndex[0]] = -1;
    }

    private void findClusterCentres(){
        clusterCenters = new ArrayList<>();
        assignments = new int[numInstances];

        //Get the cluster center estimates.
        double[] estimates = new double[numInstances];
        double sum = 0;

        for (int i = 0; i < numInstances; i++){
            estimates[i] = localDensities[i]*shortestDist[i];
            sum += estimates[i];
        }

        //Matlab commands to plot out the decision graph.
        //not very sophisticated i know
//        System.out.println("figure");
//        System.out.println("b = " + Arrays.toString(localDensities));
//        System.out.println("v = " + Arrays.toString(shortestDist));
//        System.out.println("scatter(b,v)");

        //Same as above but for estimates
//        System.out.println("figure");
//        System.out.println("e = " + Arrays.toString(estimates));
//        System.out.println("scatter([1:length(e)],sort(e))");

        //Find the indexes of the estimates sorted in ascending order
        Integer[] estIndexes = new Integer[numInstances];
        for (int i = 0; i < numInstances; i++){
            estIndexes[i] = i;
        }
        GenericTools.SortIndexAscending sort = new GenericTools.SortIndexAscending(estimates);
        Arrays.sort(estIndexes, sort);

        double mean = sum/numInstances;
        boolean threshholdFound = false;
        boolean findCutoff = false;

        for (int i = 0; i < numInstances; i++){
            //If no estimate cutoff value is set find a cutoff point
            if (clusterCenterCutoff < 0){
                findCutoff = true;
            }

            if (findCutoff){
                clusterCenterCutoff = estimates[estIndexes[i]] + mean;
            }

            //If a value above the cutoff is found set it and the following
            //points as cluster centers
            if (threshholdFound || i == numInstances-1){
                clusterCenters.add(estIndexes[i]);
                assignments[estIndexes[i]] = estIndexes[i];
            }
            else if (clusterCenterCutoff < estimates[estIndexes[i+1]]){
                threshholdFound = true;
                assignments[estIndexes[i]] = -1;
            }
            else{
                assignments[estIndexes[i]] = -1;
            }
        }
    }

    //Assigns each point to a cluster by setting each to the cluster of its
    //nearest neighbour, iterating through the sorted local densities
    private void assignClusters(){
        for (int i = 0; i < numInstances; i++){
            if (!clusterCenters.contains(sortedDensitiesIndex[i])){
                assignments[sortedDensitiesIndex[i]] = assignments[nearestNeighbours[sortedDensitiesIndex[i]]];
            }
        }

        if (cutoffOutlierSelection){
            cutoffOutliers();
        }
        else if (haloOutlierSelection){
            haloOutliers();
        }
    }

    //Sets points as not belonging to a cluster using a cutoff for its local
    //density
    private void cutoffOutliers(){
        if (outlierCutoff < 0){
            outlierCutoff = numInstances/20;
        }

        for (int i = 0; i < numInstances; i++){
            if (localDensities[i] < outlierCutoff){
                assignments[i] = -1;
            }
        }
    }

    //Sets points as not belonging to a cluster using the original
    //implementations halo method, setting points outside the found clusters
    //border as clusterless.
    private void haloOutliers(){
        if (clusterCenters.size() > 0){
            double[] border = new double[numInstances]; //larger than it should be

            for (int i = 0; i < numInstances; i++){
                for (int n = 0; n < i; n++){
                    if (assignments[i] != assignments[n] && distanceMatrix[i][n] <= distC){
                        double ldAvg = (localDensities[i] + localDensities[n])/2;

                        if (ldAvg > border[assignments[i]]) {
                            border[assignments[i]] = ldAvg;
                        }

                        if (ldAvg > border[assignments[n]]){
                            border[assignments[n]] = ldAvg;
                        }
                    }
                }
            }

            for (int i = 0; i < numInstances; i++){
                if (localDensities[i] < border[assignments[i]]){
                    assignments[i] = -1;
                }
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
        double[] cutoffs = {0.75, 1.5, 1, 4, 2, 0.3};

//        String[] datasets = {"Z:/Data/ClusteringTestDatasets/DensityPeakVector/aggregation.arff"};
//        String[] names = {"aggre"};

        boolean output = true;

        if(output){
            System.out.println("cd('Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\" +
                    "DensityPeakVector')");
            System.out.println("load('matlabCluster.mat')");
        }

        for (int i = 0; i < datasets.length; i++){
            Instances inst = DatasetLoading.loadDataNullable(datasets[i]);
            inst.setClassIndex(inst.numAttributes()-1);
            DensityPeaks dp = new DensityPeaks();
            dp.setClusterCenterCutoff(cutoffs[i]);
            dp.setGaussianKernel(true);
            dp.setHaloOutlierSelection(false);
            dp.buildClusterer(inst);

            if(output){
                System.out.println(names[i] + "c = " + Arrays.toString(dp.assignments));
                System.out.println("figure");
                System.out.println("scatter(" + names[i] + "x," + names[i] + "y,[],scatterColours(" + names[i] + "c))");
            }
        }
    }
}
