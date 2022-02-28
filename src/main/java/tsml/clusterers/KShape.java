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
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;
import weka.clusterers.NumberOfClustersRequestable;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.EigenvalueDecomposition;
import weka.core.matrix.Matrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import static utilities.ClusteringUtilities.randIndex;
import static utilities.ClusteringUtilities.zNormalise;
import static utilities.GenericTools.indexOfMax;
import static utilities.InstanceTools.deleteClassAttribute;

/**
 * Class for the KShape clustering algorithm.
 *
 * @author Matthew Middlehurst
 */
public class KShape extends EnhancedAbstractClusterer implements NumberOfClustersRequestable {

    //Paparrizos, John, and Luis Gravano.
    //"k-shape: Efficient and accurate clustering of time series."
    //Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data. ACM, 2015.

    private int k = 2;
    private int maxIterations = 100;

    private Instances centroids;

    public KShape() {
    }

    @Override
    public int numberOfClusters() {
        return k;
    }

    @Override
    public void setNumClusters(int numClusters) throws Exception {
        k = numClusters;
    }

    public void setMaxIterations(int i) {
        maxIterations = i;
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        super.buildClusterer(data);

        zNormalise(train);

        ArrayList<Attribute> atts = new ArrayList(train.numAttributes());
        for (int i = 0; i < train.numAttributes(); i++) {
            atts.add(new Attribute("att" + i));
        }

        centroids = new Instances("centroids", atts, k);
        for (int i = 0; i < k; i++) {
            centroids.add(new DenseInstance(1, new double[train.numAttributes()]));
        }

        Random rand;
        if (!seedClusterer) {
            rand = new Random();
        } else {
            rand = new Random(seed);
        }

        assignments = new double[train.numInstances()];
        //Randomly assign clusters
        for (int i = 0; i < assignments.length; i++) {
            assignments[i] = (int) Math.ceil(rand.nextDouble() * k) - 1;
        }

        SBD sbd = new SBD();

        int iterations = 0;
        double[] prevCluster = new double[train.numInstances()];
        prevCluster[0] = -1;
        //While clusters change and less than max iterations
        while (!Arrays.equals(assignments, prevCluster) && iterations < maxIterations) {
            prevCluster = Arrays.copyOf(assignments, assignments.length);

            //Select centroids
            for (int i = 0; i < k; i++) {
                centroids.set(i, shapeExtraction(train, centroids.get(i), i));
            }

            //Set each instance to the cluster of its closest centroid using shape based distance
            for (int i = 0; i < train.numInstances(); i++) {
                double minDist = Double.MAX_VALUE;

                for (int n = 0; n < k; n++) {
                    double dist = sbd.calculateDistance(centroids.get(n), train.get(i));

                    if (dist < minDist) {
                        minDist = dist;
                        assignments[i] = n;
                    }
                }
            }

            iterations++;
        }

        //Create and store an ArrayList for each cluster containing indexes of
        //points inside the cluster.
        clusters = new ArrayList[k];

        for (int i = 0; i < k; i++) {
            clusters[i] = new ArrayList();
        }

        for (int i = 0; i < train.numInstances(); i++) {
            clusters[(int) assignments[i]].add(i);
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

        zNormalise(newInst);

        double minDist = Double.MAX_VALUE;
        int closestCluster = 0;
        for (int i = 0; i < centroids.size(); ++i) {
            SBD sbd = new SBD(newInst, centroids.get(i), false);

            if (sbd.dist < minDist) {
                minDist = sbd.dist;
                closestCluster = i;
            }
        }

        return closestCluster;
    }

    private Instance shapeExtraction(Instances data, Instance centroid, int centroidNum) {
        Instances subsample = new Instances(data, 0);
        int seriesSize = centroid.numAttributes();

        double sum = 0;
        for (int i = 0; i < seriesSize; i++) {
            sum += centroid.value(i);
        }
        boolean sumZero = sum == 0;

        //Take subsample of instances in centroids cluster
        for (int i = 0; i < data.numInstances(); i++) {
            if (assignments[i] == centroidNum) {
                //If the centroid sums to 0 add full instance to the subsample
                if (sumZero) {
                    subsample.add(data.get(i));
                } else {
                    SBD sbd = new SBD(centroid, data.get(i), true);
                    subsample.add(sbd.yShift);
                }
            }
        }

        //Return instances of 0s as centroid if subsample empty
        if (subsample.numInstances() == 0) {
            return new DenseInstance(1, new double[centroid.numAttributes()]);
        }

        zNormalise(subsample);

        double[][] subsampleArray = new double[subsample.numInstances()][];

        for (int i = 0; i < subsample.numInstances(); i++) {
            subsampleArray[i] = subsample.get(i).toDoubleArray();
        }

        //Calculate eignenvectors for subsample
        Matrix matrix = new Matrix(subsampleArray);
        Matrix matrixT = matrix.transpose();

        matrix = matrixT.times(matrix);

        Matrix identity = Matrix.identity(seriesSize, seriesSize);
        Matrix ones = new Matrix(seriesSize, seriesSize, 1);
        ones = ones.times(1.0 / seriesSize);
        identity = identity.minus(ones);

        matrix = identity.times(matrix).times(identity);

        // todo If we dont add a max iterations it can fail to converge
//        EigenvalueDecomposition.maxIter = 100000;
        EigenvalueDecomposition eig = matrix.eig();
//        EigenvalueDecomposition.maxIter = -1;
        Matrix v = eig.getV();
        double[] eigVector = new double[centroid.numAttributes()];
        double[] eigVectorNeg = new double[centroid.numAttributes()];

        double eigSum = 0;
        double eigSumNeg = 0;

        int col = 0;
        while (true) {
            for (int i = 0; i < seriesSize; i++) {
                eigVector[i] = v.get(i, col);
                eigVectorNeg[i] = -eigVector[i];

                double firstVal = subsample.get(0).value(i);

                eigSum += (firstVal - eigVector[i]) * (firstVal - eigVector[i]);
                eigSumNeg += (firstVal - eigVectorNeg[i]) * (firstVal - eigVectorNeg[i]);
            }

            //Hack to move to next column if the correct values dont appear on the first one for some reason
            //I have no idea why this happens or which datasets this may happen in
            if (Math.round(eigSum) == subsample.get(0).numAttributes() &&
                    Math.round(eigSumNeg) == subsample.get(0).numAttributes()) {
                col++;
            } else {
                break;
            }
        }

        Instance newCent;
        if (eigSum < eigSumNeg) {
            newCent = new DenseInstance(1, eigVector);
        } else {
            newCent = new DenseInstance(1, eigVectorNeg);
        }

        //Normalise and return eigenvector as new centroid
        zNormalise(newCent);

        return newCent;
    }

    public static void main(String[] args) throws Exception {
        String dataset = "Trace";
        Instances inst = DatasetLoading.loadDataNullable("D:\\CMP Machine Learning\\Datasets\\UnivariateARFF\\" + dataset + "/" +
                dataset + "_TRAIN.arff");
        Instances inst2 = DatasetLoading.loadDataNullable("D:\\CMP Machine Learning\\Datasets\\UnivariateARFF\\" + dataset + "/" +
                dataset + "_TEST.arff");
        inst.setClassIndex(inst.numAttributes() - 1);
        inst.addAll(inst2);

        KShape k = new KShape();
        k.setSeed(0);
        k.k = inst.numClasses();
        k.buildClusterer(inst);

        System.out.println(k.clusters.length);
        System.out.println(Arrays.toString(k.assignments));
        System.out.println(Arrays.toString(k.clusters));
        System.out.println(randIndex(k.assignments, inst));
    }

    //Class for calculating Shape Based Distance
    public static class SBD {

        private double dist;
        private Instance yShift;

        private FastFourierTransformer fft;

        public SBD() {
        }

        private SBD(Instance first, Instance second, boolean calcShift) {
            calculateDistance(first, second, calcShift);
        }

        public double[][] createDistanceMatrix(Instances data){
            double[][] distMatrix = new double[data.numInstances()][];

            for (int i = 0; i < data.numInstances(); i++){
                distMatrix[i] = new double[data.numInstances()];
                Instance first = data.get(i);

                for (int n = 0; n < data.numInstances(); n++){
                    distMatrix[i][n] = calculateDistance(first, data.get(n));
                }
            }

            return distMatrix;
        }

        public double[][] createBottomHalfDistanceMatrix(Instances data){
            double[][] distMatrix = new double[data.numInstances()][];

            for (int i = 0; i < data.numInstances(); i++){
                distMatrix[i] = new double[i + 1];
                Instance first = data.get(i);

                for (int n = 0; n < i; n++){
                    distMatrix[i][n] = calculateDistance(first, data.get(n));
                }
            }

            return distMatrix;
        }

        public double calculateDistance(Instance first, Instance second) {
            calculateDistance(first, second, false);
            return dist;
        }

        private void calculateDistance(Instance first, Instance second, boolean calcShift) {
            int oldLength = first.numAttributes();
            int oldLengthY = second.numAttributes();
            int maxLength = Math.max(oldLength, oldLengthY);

            int nfft = (int) Math.pow(2.0, (int) Math.ceil(Math.log(maxLength) / Math.log(2)));

            //FFT and IFFT
            fft = new FastFourierTransformer(DftNormalization.STANDARD);

            Complex[] firstC = fft(first, oldLength, nfft);
            Complex[] secondC = fft(second, oldLengthY, nfft);
            for (int i = 0; i < nfft; i++) {
                firstC[i] = firstC[i].multiply(secondC[i].conjugate());
            }

            firstC = fft.transform(firstC, TransformType.INVERSE);

            //Calculate NCCc values
            double firstNorm = sumSquare(first);
            double secondNorm = sumSquare(second);
            double norm = Math.sqrt(firstNorm * secondNorm);

            double[] ncc = new double[oldLength * 2 - 1];
            int idx = 0;

            for (int i = nfft - oldLength + 1; i < nfft; i++) {
                ncc[idx++] = firstC[i].getReal() / norm;
            }

            for (int i = 0; i < oldLength; i++) {
                ncc[idx++] = firstC[i].getReal() / norm;
            }

            double maxValue = 0;
            int shift = -1;
            //Largest NCCc value and index
            for (int i = 0; i < ncc.length; i++) {
                if (ncc[i] > maxValue) {
                    maxValue = ncc[i];
                    shift = i;
                }
            }

            dist = 1 - maxValue;

            //Create y', shifting the second instance in a direction and padding with 0s
            if (calcShift) {
                shift -= maxLength - 1;

                yShift = new DenseInstance(1, new double[oldLengthY]);

                if (shift >= 0) {
                    for (int i = 0; i < oldLengthY - shift; i++) {
                        yShift.setValue(i + shift, second.value(i));
                    }
                } else {
                    for (int i = 0; i < oldLengthY + shift; i++) {
                        yShift.setValue(i, second.value(i - shift));
                    }
                }
            }
        }

        //Run FFT and return array of complex numbers
        private Complex[] fft(Instance inst, int oldLength, int nfft) {
            Complex[] complex = new Complex[nfft];

            for (int i = 0; i < oldLength; i++) {
                complex[i] = new Complex(inst.value(i), 0);
            }

            for (int i = oldLength; i < nfft; i++) {
                complex[i] = new Complex(0, 0);
            }

            return fft.transform(complex, TransformType.FORWARD);
        }

        private double sumSquare(Instance inst) {
            double sum = 0;
            for (int i = 0; i < inst.numAttributes(); i++) {
                sum += inst.value(i) * inst.value(i);
            }
            return sum;
        }
    }
}
