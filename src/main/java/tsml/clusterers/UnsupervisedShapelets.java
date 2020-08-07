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
package tsml.clusterers;

import java.util.ArrayList;
import java.util.Arrays;

import experiments.data.DatasetLoading;
import weka.core.Instance;
import weka.core.Instances;
import machine_learning.clusterers.KMeans;

import static utilities.ClusteringUtilities.randIndex;
import static utilities.InstanceTools.deleteClassAttribute;
import static utilities.InstanceTools.toWekaInstances;

/**
 * Class for the UnsupervisedShapelets clustering algorithm.
 *
 * @author Matthew Middlehurst
 */
public class UnsupervisedShapelets extends AbstractTimeSeriesClusterer{

    //Zakaria, Jesin, Abdullah Mueen, and Eamonn Keogh.
    //"Clustering time series using unsupervised-shapelets."
    //2012 IEEE 12th International Conference on Data Mining. IEEE, 2012.

    private int k = 2;
    private int numFolds = 20;
    private int seed = Integer.MIN_VALUE;

    private ArrayList<UShapelet> shapelets;
    private int numInstances;

    public UnsupervisedShapelets(){}

    @Override
    public int numberOfClusters(){
        return k;
    }

    public void setNumberOfClusters(int n){ k = n; }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        if (copyInstances){
            data = new Instances(data);
        }

        deleteClassAttribute(data);

        extractUShapelets(data);
        clusterData(data);
    }

    private void extractUShapelets(Instances data){
        int[] shapeletLengths = {25, 50};

        if (data.numAttributes() < 50){
            shapeletLengths = new int[]{data.numAttributes()/2};
        }

        shapelets = new ArrayList();
        numInstances = data.size();
        Instance inst = data.firstInstance();
        boolean finished = false;

        while (!finished){
            ArrayList<UShapelet> shapeletCandidates = new ArrayList();

            //Finds all candidate shapelets on the selected instance
            for (int i = 0; i < shapeletLengths.length; i++){
                for (int n = 0; n < inst.numAttributes() - shapeletLengths[i]; n++){
                    UShapelet candidate = new UShapelet(n, shapeletLengths[i], inst);
                    candidate.computeGap(data);
                    shapeletCandidates.add(candidate);
                }
            }

            double maxGap = -1;
            int maxGapIndex = -1;

            //Finds the shapelet with the highest gap value
            for (int i = 0; i < shapeletCandidates.size(); i++){
                if (shapeletCandidates.get(i).gap > maxGap){
                    maxGap = shapeletCandidates.get(i).gap;
                    maxGapIndex = i;
                }
            }

            //Adds the shapelet with the best gap value to the pool of shapelets
            UShapelet best = shapeletCandidates.get(maxGapIndex);
            shapelets.add(best);

            double[] distances = best.computeDistances(data);
            ArrayList<Double> lesserDists = new ArrayList();
            double maxDist = -1;
            int maxDistIndex = -1;

            //Finds the instance with the max dist to the shapelet and all with a dist lower than the distance used
            //to generate the gap value
            for (int i = 0; i < distances.length; i++){
                if (distances[i] < best.dt){
                    lesserDists.add(distances[i]);
                }
                else if (distances[i] > maxDist){
                    maxDist = distances[i];
                    maxDistIndex = i;
                }
            }

            //Use max dist instance to generate new shapelet and remove low distance instances
            if (lesserDists.size() == 1){
                finished = true;
            }
            else{
                inst = data.get(maxDistIndex);

                double mean = mean(lesserDists);
                double cutoff = mean + standardDeviation(lesserDists, mean);

                Instances newData = new Instances(data, 0);

                for (int i = 0; i < data.numInstances(); i++){
                    if (distances[i] >= cutoff){
                        newData.add(data.get(i));
                    }
                }

                data = newData;

                if (data.size() == 1){
                    finished = true;
                }
            }
        }
    }

    private void clusterData(Instances data) throws Exception{
        Instances distanceMap;

        int[][] foldClusters = new int[shapelets.size()][];
        double[][] distanceMatrix = new double[numInstances][1];
        double minRandIndex = 1;
        int minIndex = -1;

        //Create a distance matrix by calculating the distance of shapelet i and previous shapelets to each time series
        for (int i = 0; i < shapelets.size(); i++){
            UShapelet shapelet = shapelets.get(i);
            double[] distances = shapelet.computeDistances(data);
            double minDist = Double.MAX_VALUE;

            for (int n = 0; n < numInstances; n++) {
                distanceMatrix[n] = Arrays.copyOf(distanceMatrix[n], i+1);
                distanceMatrix[n][i] = distances[n];
            }

            distanceMap = toWekaInstances(distanceMatrix);

            //Build multiple kmeans clusterers using the one with the smallest squared distance
            for (int n = 0; n < numFolds; n++){
                KMeans kmeans = new KMeans();
                kmeans.setNumberOfClusters(k);
                kmeans.setNormaliseData(false);
                kmeans.setFindBestK(false);
                kmeans.setRefinedInitialMedoids(false);
                kmeans.setSeed(seed+n);
                kmeans.buildClusterer(distanceMap);

                double dist = kmeans.clusterSquaredDistance(distanceMap);

                if (dist < minDist){
                    minDist = dist;
                    foldClusters[i] = kmeans.getAssignments();
                }
            }

            double randIndex = 1;

            //If the rand index of this output of clusters compared to the previous one is greater than the current best
            //use this output of clusters
            if (i > 0){
                randIndex = 1-randIndex(foldClusters[i-1],foldClusters[i]);
            }

            if (randIndex < minRandIndex){
                minRandIndex = randIndex;
                minIndex = i;
            }
        }

        assignments = foldClusters[minIndex];

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
    }

    private double mean(ArrayList<Double> dists){
        double meanSum = 0;

        for (int i = 0; i < dists.size(); i++){
            meanSum += dists.get(i);
        }

        return meanSum / dists.size();
    }

    private double standardDeviation(ArrayList<Double> dists, double mean){
        double sum = 0;
        double temp;

        for (int i = 0; i < dists.size(); i++){
            temp = dists.get(i) - mean;
            sum += temp * temp;
        }

        double meanOfDiffs = sum/dists.size();
        return Math.sqrt(meanOfDiffs);
    }

    public static void main(String[] args) throws Exception{
        String dataset = "Trace";
        Instances inst = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset + "/" +
                dataset + "_TRAIN.arff");
        Instances inst2 = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset + "/" +
                dataset + "_TEST.arff");
        inst.setClassIndex(inst.numAttributes()-1);
        inst.addAll(inst2);

        UnsupervisedShapelets us = new UnsupervisedShapelets();
        us.seed = 0;
        us.k = inst.numClasses();
        us.buildClusterer(inst);

        System.out.println(us.clusters.length);
        System.out.println(Arrays.toString(us.assignments));
        System.out.println(Arrays.toString(us.clusters));
        System.out.println(randIndex(us.assignments, inst));
    }

    //Class for a single Unsupervised Shapelet with methods to calculate distance to time series and the gap value
    private class UShapelet{

        //Where the shapelet starts
        int startPoint;
        //Length of the shapelet
        int length;
        //Series the shapelet is extracted from
        double[] series;

        double gap = 0;
        double dt = 0;

        UShapelet(int startPoint, int length, Instance inst){
            this.startPoint = startPoint;
            this.length = length;
            this.series = inst.toDoubleArray();
        }

        //finds the highest gap value and corresponding distance for this shapelet on the input dataset
        void computeGap(Instances data){
            double[] sortedDistances = computeDistances(data);
            Arrays.sort(sortedDistances);

            for (int i = 0; i < sortedDistances.length-1; i++){
                double dist = (sortedDistances[i] + sortedDistances[i+1])/2;

                ArrayList<Double> lesserDists = new ArrayList();
                ArrayList<Double> greaterDists = new ArrayList();

                //separate instance distances based on whether they are greater or less than the current dist
                for (int n = 0; n < sortedDistances.length; n++){
                    if (sortedDistances[n] < dist){
                        lesserDists.add(sortedDistances[n]);
                    }
                    else{
                        greaterDists.add(sortedDistances[n]);
                    }
                }

                double ratio = lesserDists.size()/greaterDists.size();

                if (1.0/k < ratio){
                    double lesserMean = mean(lesserDists);
                    double greaterMean = mean(greaterDists);
                    double lesserStdev = standardDeviation(lesserDists, lesserMean);
                    double greaterStdev = standardDeviation(greaterDists, greaterMean);

                    //gap value for this distance
                    double gap = greaterMean - greaterStdev - (lesserMean + lesserStdev);

                    if (gap > this.gap){
                        this.gap = gap;
                        this.dt = dist;
                    }
                }
            }
        }

        //Lowest euclidean distance of the shapelet to each instance in the input dataset
        double[] computeDistances(Instances data){
            double[] distances = new double[data.numInstances()];
            double[] shapelet = zNormalise();

            for (int i = 0; i < data.numInstances(); i++){
                Instance inst = data.get(i);
                distances[i] = Double.MAX_VALUE;
                UShapelet subseries = new UShapelet(0, length, inst);

                //Sliding window calculating distance of each section of the series to the shapelet
                for (int n = 0; n < inst.numAttributes() - length; n++){
                    subseries.startPoint = n;
                    double dist = euclideanDistance(shapelet, subseries.zNormalise());

                    if (dist < distances[i]){
                        distances[i] = dist;
                    }
                }
            }

            double normaliser = Math.sqrt(length);

            for (int i = 0; i < distances.length; i++){
                distances[i] /= normaliser;
            }

            return distances;
        }

        //return the shapelet using the series, start point and shapelet length
        double[] zNormalise(){
            double meanSum = 0;

            for (int i = startPoint; i < startPoint + length; i++){
                meanSum += series[i];
            }

            double mean = meanSum / length;

            double stdevSum = 0;
            double temp;

            for (int i = startPoint; i < startPoint + length; i++){
                temp = series[i] - mean;
                stdevSum += temp * temp;
            }

            double stdev = Math.sqrt(stdevSum/length);

            double[] output = new double[length];

            if (stdev != 0){
                for (int i = startPoint; i < startPoint + length; i++){
                    output[i - startPoint] = (series[i] - mean) / stdev;
                }
            }

            return output;
        }

        private double euclideanDistance(double[] series1, double[] series2){
            double dist = 0;

            for(int i = 0; i < series1.length; i++){
                double n = series1[i] - series2[i];
                dist += n*n;
            }

            return Math.sqrt(dist);
        }
    }
}