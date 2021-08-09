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

package tsml.clusterers;

import experiments.data.DatasetLoading;
import machine_learning.clusterers.KMeans;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;
import weka.clusterers.NumberOfClustersRequestable;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static utilities.ArrayUtilities.*;
import static utilities.ClusteringUtilities.randIndex;
import static utilities.GenericTools.min;
import static utilities.InstanceTools.deleteClassAttribute;
import static utilities.InstanceTools.toWekaInstances;

/**
 * Class for the UnsupervisedShapelets clustering algorithm.
 *
 * @author Matthew Middlehurst
 */
public class UnsupervisedShapelets extends EnhancedAbstractClusterer implements NumberOfClustersRequestable {

    //Zakaria, Jesin, Abdullah Mueen, and Eamonn Keogh.
    //"Clustering time series using unsupervised-shapelets."
    //2012 IEEE 12th International Conference on Data Mining. IEEE, 2012.

    private int k = 2;
    private boolean useKMeans = true;
    private int numKMeansFolds = 20;
    private int[] shapeletLengths = {50};
    private boolean exhaustiveSearch = false;
    private double randomSearchProportion = -1;

    private ArrayList<UShapelet> shapelets;
    private KMeans shapeletClusterer;
    private Instances header;
    private int numShapeletsToUse;
    private int numInstances;
    private double firstGap;

    public UnsupervisedShapelets() {
    }

    @Override
    public int numberOfClusters() {
        return k;
    }

    @Override
    public void setNumClusters(int numClusters) throws Exception {
        k = numClusters;
    }

    public void setUseKMeans(boolean b){
        useKMeans = b;
    }

    public void setNumKMeansFolds(int i) {
        numKMeansFolds = i;
    }

    public void setShapeletLengths(int[] arr){
        shapeletLengths = arr;
    }

    public void setExhaustiveSearch(boolean b){
        exhaustiveSearch = b;
    }

    public void setRandomSearchProportion(double d){
        randomSearchProportion = d;
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        super.buildClusterer(data);

        extractUShapelets(train);
        clusterData(train);
    }

    @Override
    public int clusterInstance(Instance inst) throws Exception {
        Instance newInst = copyInstances ? new DenseInstance(inst) : inst;
        deleteClassAttribute(newInst);

        if (useKMeans) {
            Instance shapeletDists = new DenseInstance(numShapeletsToUse);
            for (int i = 0; i < numShapeletsToUse; i++) {
                shapeletDists.setValue(i, shapelets.get(i).computeDistance(inst));
            }
            shapeletDists.setDataset(header);

            return shapeletClusterer.clusterInstance(shapeletDists);
        }
        else {
            double minDist = Double.MAX_VALUE;
            int minIdx = -1;

            for (int i = 0; i < shapelets.size(); i++) {
                double dist = shapelets.get(i).computeDistance(inst);

                if (dist < minDist){
                    minDist = dist;
                    minIdx = i;
                }
            }

            return minIdx;
        }
    }

    private void extractUShapelets(Instances data) {
        if (data.numAttributes() / 2 < min(shapeletLengths)) {
            shapeletLengths = new int[]{data.numAttributes() / 2};
        }

        ArrayList<Integer> indicies = null;
        if (!useKMeans) {
            assignments = new double[data.numInstances()];

            indicies = new ArrayList<>(assignments.length);
            for (int i = assignments.length - 1; i >= 0; i--){
                indicies.add(i);
            }
        }

        Random rand;
        if (!seedClusterer) {
            rand = new Random();
        } else {
            rand = new Random(seed);
        }

        shapelets = new ArrayList();
        numInstances = data.size();
        Instance inst = data.firstInstance();

        boolean finished = false;
        int iteration = 0;
        while (!finished) {
            ArrayList<UShapelet> shapeletCandidates = new ArrayList();

            if (exhaustiveSearch){
                //Finds all candidate shapelets on all instances
                for (int shapeletLength : shapeletLengths) {
                    for (int j = 0; j < data.numInstances(); j++) {
                        inst = data.get(j);
                        for (int n = 0; n < inst.numAttributes() - shapeletLength; n++) {
                            UShapelet candidate = new UShapelet(n, shapeletLength, inst);
                            candidate.computeGap(data);
                            shapeletCandidates.add(candidate);
                        }
                    }
                }
            }
            else if (randomSearchProportion > 0){
                //Finds all candidate shapelets on a random selection of instances
                int seriesToSelect = (int) Math.ceil(data.numInstances() * randomSearchProportion);
                ArrayList<Integer> randomIndicies = new ArrayList<>(data.numInstances());
                for (int i = 0; i < data.numInstances(); i++){
                    randomIndicies.add(i);
                }

                for (int shapeletLength : shapeletLengths) {
                    for (int j = 0; j < seriesToSelect; j++) {
                        inst = data.get(randomIndicies.remove(rand.nextInt(randomIndicies.size())));
                        for (int n = 0; n < inst.numAttributes() - shapeletLength; n++) {
                            UShapelet candidate = new UShapelet(n, shapeletLength, inst);
                            candidate.computeGap(data);
                            shapeletCandidates.add(candidate);
                        }
                    }
                }
            }
            else {
                //Finds all candidate shapelets on the selected instance
                for (int shapeletLength : shapeletLengths) {
                    for (int n = 0; n < inst.numAttributes() - shapeletLength; n++) {
                        UShapelet candidate = new UShapelet(n, shapeletLength, inst);
                        candidate.computeGap(data);
                        shapeletCandidates.add(candidate);
                    }
                }
            }

            double maxGap = -1;
            int maxGapIndex = -1;
            //Finds the shapelet with the highest gap value
            for (int i = 0; i < shapeletCandidates.size(); i++) {
                if (shapeletCandidates.get(i).gap > maxGap) {
                    maxGap = shapeletCandidates.get(i).gap;
                    maxGapIndex = i;
                }
            }

            if (!useKMeans && iteration == 0){
                firstGap = maxGap;
            }

            if (!useKMeans && maxGap < firstGap / 2){
                break;
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
            for (int i = 0; i < distances.length; i++) {
                if (distances[i] < best.dt) {
                    lesserDists.add(distances[i]);
                } else if (distances[i] > maxDist) {
                    maxDist = distances[i];
                    maxDistIndex = i;
                }
            }

            //Use max dist instance to generate new shapelet and remove low distance instances
            if (lesserDists.size() == 1) {
                finished = true;
            } else {
                inst = data.get(maxDistIndex);

                double mean = mean(lesserDists);
                double cutoff = mean + standardDeviation(lesserDists, mean);

                Instances newData = new Instances(data, 0);
                for (int i = 0; i < data.numInstances(); i++) {
                    if (distances[i] >= cutoff) {
                        newData.add(data.get(i));
                    }
                    else if (!useKMeans){
                        assignments[indicies.remove(data.numInstances() - i)] = iteration;
                    }
                }

                data = newData;
                if (data.size() == 1) {
                    finished = true;
                }
                else{
                    iteration++;
                }
            }
        }

        if (!useKMeans){
            for (int idx: indicies){
                assignments[idx] = -1;
            }
        }
    }

    private void clusterData(Instances data) throws Exception {
        if (useKMeans) {
            Instances distanceMap;
            double[][] foldClusters = new double[shapelets.size()][];
            double[][] distanceMatrix = new double[numInstances][1];
            double minRandIndex = 1;
            KMeans bestClusterer = null;

            //Create a distance matrix by calculating the distance of shapelet i and previous shapelets to each time
            //series
            for (int i = 0; i < shapelets.size(); i++) {
                UShapelet shapelet = shapelets.get(i);
                double[] distances = shapelet.computeDistances(data);
                double minDist = Double.MAX_VALUE;

                for (int n = 0; n < numInstances; n++) {
                    distanceMatrix[n] = Arrays.copyOf(distanceMatrix[n], i + 1);
                    distanceMatrix[n][i] = distances[n];
                }

                distanceMap = toWekaInstances(distanceMatrix);

                //Build multiple kmeans clusterers using the one with the smallest squared distance
                for (int n = 0; n < numKMeansFolds; n++) {
                    KMeans kmeans = new KMeans();
                    kmeans.setNumClusters(k);
                    kmeans.setNormaliseData(false);
                    kmeans.setCopyInstances(false);
                    if (seedClusterer)
                        kmeans.setSeed(seed + (n + 7) * (i + 7));
                    kmeans.buildClusterer(distanceMap);

                    double dist = kmeans.clusterSquaredDistance(distanceMap);

                    if (dist < minDist) {
                        minDist = dist;
                        foldClusters[i] = kmeans.getAssignments();
                        bestClusterer = kmeans;
                    }
                }

                double randIndex = 1;

                //If the rand index of this output of clusters compared to the previous one is greater than the current
                //best use this output of clusters
                if (i > 0) {
                    randIndex = 1 - randIndex(foldClusters[i - 1], foldClusters[i]);
                }

                if (randIndex < minRandIndex) {
                    minRandIndex = randIndex;
                    shapeletClusterer = bestClusterer;
                    header = new Instances(distanceMap, 0);
                    numShapeletsToUse = i;
                }
            }

            assignments = foldClusters[numShapeletsToUse];
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
        }
        else{
            List<Double> u = unique(assignments);
            clusters = new ArrayList[u.size()];

            for (int i = 0; i < clusters.length; i++) {
                clusters[i] = new ArrayList();
            }

            for (int i = 0; i < numInstances; i++) {
                for (int n = 0; n < clusters.length; n++) {
                    if (n == assignments[i]) {
                        clusters[n].add(i);
                        break;
                    }
                }
            }
        }
    }

    private double mean(ArrayList<Double> dists) {
        double meanSum = 0;

        for (Double dist : dists) {
            meanSum += dist;
        }

        return meanSum / dists.size();
    }

    private double standardDeviation(ArrayList<Double> dists, double mean) {
        double sum = 0;
        double temp;

        for (Double dist : dists) {
            temp = dist - mean;
            sum += temp * temp;
        }

        double meanOfDiffs = sum / dists.size();
        return Math.sqrt(meanOfDiffs);
    }

    public static void main(String[] args) throws Exception {
        String dataset = "Trace";
        Instances inst = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset + "/" +
                dataset + "_TRAIN.arff");
        Instances inst2 = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset + "/" +
                dataset + "_TEST.arff");
        inst.setClassIndex(inst.numAttributes() - 1);
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
    private class UShapelet {

        //Where the shapelet starts
        int startPoint;
        //Length of the shapelet
        int length;
        //Series the shapelet is extracted from
        double[] series;

        double gap = 0;
        double dt = 0;

        UShapelet(int startPoint, int length, Instance inst) {
            this.startPoint = startPoint;
            this.length = length;
            this.series = inst.toDoubleArray();
        }

        //finds the highest gap value and corresponding distance for this shapelet on the input dataset
        void computeGap(Instances data) {
            double[] sortedDistances = computeDistances(data);
            Arrays.sort(sortedDistances);

            for (int i = 0; i < sortedDistances.length - 1; i++) {
                double dist = (sortedDistances[i] + sortedDistances[i + 1]) / 2;

                ArrayList<Double> lesserDists = new ArrayList();
                ArrayList<Double> greaterDists = new ArrayList();

                //separate instance distances based on whether they are greater or less than the current dist
                for (double sortedDistance : sortedDistances) {
                    if (sortedDistance < dist) {
                        lesserDists.add(sortedDistance);
                    } else {
                        greaterDists.add(sortedDistance);
                    }
                }

                double ratio = (double) lesserDists.size() / greaterDists.size();

                if (1.0 / k < ratio) {
                    double lesserMean = mean(lesserDists);
                    double greaterMean = mean(greaterDists);
                    double lesserStdev = standardDeviation(lesserDists, lesserMean);
                    double greaterStdev = standardDeviation(greaterDists, greaterMean);

                    //gap value for this distance
                    double gap = greaterMean - greaterStdev - (lesserMean + lesserStdev);

                    if (gap > this.gap) {
                        this.gap = gap;
                        this.dt = dist;
                    }
                }
            }
        }

        //Lowest euclidean distance of the shapelet to each instance in the input dataset
        double[] computeDistances(Instances data) {
            double[] distances = new double[data.numInstances()];
            double[] shapelet = zNormalise();

            double sumy = sum(shapelet);
            double sumy2 = sumPow2(shapelet);

            int nfft = (int) Math.pow(2.0, (int) Math.ceil(Math.log(data.numAttributes()) / Math.log(2)));
            Complex[] yfft = new Complex[nfft];
            for (int n = 0; n < nfft; n++) {
                if (n < length)
                    yfft[n] = new Complex(shapelet[length - n - 1], 0);
                else
                    yfft[n] = new Complex(0, 0);
            }

            FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
            yfft = fft.transform(yfft, TransformType.FORWARD);

            for (int i = 0; i < data.numInstances(); i++) {
                distances[i] = computeDistance(data.get(i).toDoubleArray(), sumy, sumy2, nfft, fft, yfft);
            }

            return distances;
        }

        double computeDistance(Instance data) {
            double[] shapelet = zNormalise();

            double sumy = sum(shapelet);
            double sumy2 = sumPow2(shapelet);

            int nfft = (int) Math.pow(2.0, (int) Math.ceil(Math.log(data.numAttributes()) / Math.log(2)));
            Complex[] yfft = new Complex[nfft];
            for (int n = 0; n < nfft; n++) {
                if (n < length)
                    yfft[n] = new Complex(shapelet[length - n - 1], 0);
                else
                    yfft[n] = new Complex(0, 0);
            }

            FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
            yfft = fft.transform(yfft, TransformType.FORWARD);

            return computeDistance(data.toDoubleArray(), sumy, sumy2, nfft, fft, yfft);
        }

        double computeDistance(double[] inst, double sumy, double sumy2, int nfft, FastFourierTransformer fft,
                               Complex[] yfft) {
            Complex[] xfft = new Complex[nfft];
            for (int n = 0; n < nfft; n++) {
                if (n < inst.length)
                    xfft[n] = new Complex(inst[n], 0);
                else
                    xfft[n] = new Complex(0, 0);
            }

            xfft = fft.transform(xfft, TransformType.FORWARD);

            Complex[] zfft = new Complex[nfft];
            for (int n = 0; n < nfft; n++) {
                zfft[n] = xfft[n].multiply(yfft[n]);
            }

            zfft = fft.transform(zfft, TransformType.INVERSE);

            double[] cumsumx = cumsum(inst);
            double[] cumsumx2 = cumsumPow2(inst);

            double[] dists = new double[inst.length - length];
            for (int i = 0; i < dists.length; i++){
                double sumx = cumsumx[i + length] - cumsumx[i];
                double sumx2 = cumsumx2[i + length] - cumsumx2[i];
                double meanx = sumx / length;
                double sigmax2 = sumx2 / length - Math.pow(meanx, 2);
                double sigmax = Math.sqrt(sigmax2);

                dists[i] = (sumx2 - 2 * sumx * meanx + length * Math.pow(meanx, 2)) / sigmax2 - 2 *
                        (zfft[i + length].getReal() - sumy * meanx) / sigmax + sumy2;
            }

            return Math.sqrt(min(dists)) / Math.sqrt(length);
        }

        //return the shapelet using the series, start point and shapelet length
        double[] zNormalise() {
            double meanSum = 0;

            for (int i = startPoint; i < startPoint + length; i++) {
                meanSum += series[i];
            }

            double mean = meanSum / length;

            double stdevSum = 0;
            double temp;

            for (int i = startPoint; i < startPoint + length; i++) {
                temp = series[i] - mean;
                stdevSum += temp * temp;
            }

            double stdev = Math.sqrt(stdevSum / length);

            double[] output = new double[length];

            if (stdev != 0) {
                for (int i = startPoint; i < startPoint + length; i++) {
                    output[i - startPoint] = (series[i] - mean) / stdev;
                }
            }

            return output;
        }
    }
}