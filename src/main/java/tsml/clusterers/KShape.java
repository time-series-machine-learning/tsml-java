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
import java.util.Random;

import experiments.data.DatasetLoading;
import weka.core.Instance;
import weka.core.Instances;
import tsml.transformers.FFT;
import tsml.transformers.FFT.Complex;
import static tsml.transformers.FFT.MathsPower2;

import static utilities.ClusteringUtilities.randIndex;
import static utilities.ClusteringUtilities.zNormalise;
import static utilities.InstanceTools.deleteClassAttribute;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.matrix.EigenvalueDecomposition;
import weka.core.matrix.Matrix;

/**
 * Class for the KShape clustering algorithm.
 *
 * @author Matthew Middlehurst
 */
public class KShape extends AbstractTimeSeriesClusterer {

    //Paparrizos, John, and Luis Gravano.
    //"k-shape: Efficient and accurate clustering of time series."
    //Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data. ACM, 2015.

    private int k = 2;
    private int seed = Integer.MIN_VALUE;

    private Instances centroids;

    public KShape(){}

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
        zNormalise(data);

        ArrayList<Attribute> atts = new ArrayList(data.numAttributes());

        for (int i = 0; i < data.numAttributes(); i++){
            atts.add(new Attribute("att" + i));
        }

        centroids = new Instances("centroids", atts, k);

        for (int i = 0; i < k; i++) {
            centroids.add(new DenseInstance(1, new double[data.numAttributes()]));
        }

        Random rand;

        if (seed == Integer.MIN_VALUE){
            rand = new Random();
        }
        else{
            rand = new Random(seed);
        }

        int iterations = 0;
        assignments = new int[data.numInstances()];

        //Randomly assign clusters
        for (int i = 0; i < assignments.length; i++){
            assignments[i] = (int)Math.ceil(rand.nextDouble()*k)-1;
        }

        int[] prevCluster = new int[data.numInstances()];
        prevCluster[0] = -1;

        //While clusters change and less than max iterations
        while (!Arrays.equals(assignments, prevCluster) && iterations < 100){
            prevCluster = Arrays.copyOf(assignments, assignments.length);

            //Select centroids
            for (int i = 0; i < k; i ++){
                centroids.set(i, shapeExtraction(data, centroids.get(i), i));
            }

            //Set each instance to the cluster of its closest centroid using shape based distance
            for (int i = 0; i < data.numInstances(); i++){
                double minDist = Double.MAX_VALUE;

                for (int n = 0; n < k; n++){
                    SBD sbd = new SBD(centroids.get(n), data.get(i), false);

                    if (sbd.dist < minDist){
                        minDist = sbd.dist;
                        assignments[i] = n;
                    }
                }
            }

            iterations++;
        }

        //Create and store an ArrayList for each cluster containing indexes of
        //points inside the cluster.
        clusters = new ArrayList[k];

        for (int i = 0; i < k; i++){
            clusters[i] = new ArrayList();
        }

        for (int i = 0; i < data.numInstances(); i++){
            for (int n = 0; n < k; n++){
                if(n == assignments[i]){
                    clusters[n].add(i);
                    break;
                }
            }
        }
    }

    private Instance shapeExtraction(Instances data, Instance centroid, int centroidNum) throws Exception {
        Instances subsample = new Instances(data, 0);
        int seriesSize = centroid.numAttributes();

        double sum = 0;
        for (int i = 0; i < seriesSize; i++){
            sum += centroid.value(i);
        }
        boolean sumZero = sum == 0;

        //Take subsample of instances in centroids cluster
        for (int i = 0; i < data.numInstances(); i++){
            if (assignments[i] == centroidNum){
                //If the centroid sums to 0 add full instance to the subsample
                if (sumZero){
                    subsample.add(data.get(i));
                }
                else{
                    SBD sbd = new SBD(centroid, data.get(i), true);
                    subsample.add(sbd.yShift);
                }
            }
        }

        //Return instances of 0s as centroid if subsample empty
        if (subsample.numInstances() == 0){
            return new DenseInstance(1, new double[centroid.numAttributes()]);
        }

        zNormalise(subsample);

        double[][] subsampleArray = new double[subsample.numInstances()][];

        for (int i = 0; i < subsample.numInstances(); i++){
            subsampleArray[i] = subsample.get(i).toDoubleArray();
        }

        //Calculate eignenvectors for subsample
        Matrix matrix = new Matrix(subsampleArray);
        Matrix matrixT = matrix.transpose();

        matrix = matrixT.times(matrix);

        Matrix identity = Matrix.identity(seriesSize, seriesSize);
        Matrix ones = new Matrix(seriesSize, seriesSize, 1);
        ones = ones.times(1.0/seriesSize);
        identity = identity.minus(ones);

        matrix = identity.times(matrix).times(identity);

        EigenvalueDecomposition eig = matrix.eig();
        Matrix v = eig.getV();
        double[] eigVector = new double[centroid.numAttributes()];
        double[] eigVectorNeg = new double[centroid.numAttributes()];

        double eigSum = 0;
        double eigSumNeg = 0;

        int col = 0;
        while(true) {
            for (int i = 0; i < seriesSize; i++) {
                eigVector[i] = v.get(i, col);
                eigVectorNeg[i] = -eigVector[i];

                double firstVal = subsample.get(0).value(i);

                eigSum += (firstVal - eigVector[i]) * (firstVal - eigVector[i]);
                eigSumNeg += (firstVal - eigVectorNeg[i]) * (firstVal - eigVectorNeg[i]);
            }

            //Hack to move to next column if the correct values dont appear on the first one for some reason
            //I have no idea why this happens or which datasets this may happen in
            if (Math.round(eigSum) == subsample.get(0).numAttributes() && Math.round(eigSumNeg) == subsample.get(0).numAttributes()){
                col++;
                System.err.println("Possible eigenvalue error, moving onto next column. Look into why this happens.");
            }
            else{
                break;
            }
        }

        Instance newCent;

        if (eigSum < eigSumNeg){
            newCent = new DenseInstance(1, eigVector);
        }
        else{
            newCent = new DenseInstance(1, eigVectorNeg);
        }

        //Normalise and return eigenvector as new centroid
        zNormalise(newCent);

        return newCent;
    }

    public static void main(String[] args) throws Exception{
//        double[] d = {1,2,3,4,5,6,7,8,9,10};
//        DenseInstance inst1 = new DenseInstance(1, d);
//
//        double[] d2 = {-1,-1,-1,1,1,1,2,2,2,2,3,3,3};
//        DenseInstance inst2 = new DenseInstance(1, d2);
//
//        ArrayList<Attribute> atts = new ArrayList();
//        for (int i = 0; i < d2.length; i++){
//            atts.add(new Attribute("att" + i));
//        }
//        Instances data = new Instances("test", atts, 0);
//        inst1.setDataset(data);
//        inst2.setDataset(data);
//
//        SBD sbd = new SBD(inst1, inst2);
//
//        System.out.println(sbd.dist);
//        System.out.println(sbd.yShift);

        String dataset = "Trace";
        Instances inst = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset + "/" +
                dataset + "_TRAIN.arff");
        Instances inst2 = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset + "/" +
                dataset + "_TEST.arff");
        inst.setClassIndex(inst.numAttributes()-1);
        inst.addAll(inst2);

        KShape k = new KShape();
        k.seed = 0;
        k.k = inst.numClasses();
        k.buildClusterer(inst);

        System.out.println(k.clusters.length);
        System.out.println(Arrays.toString(k.assignments));
        System.out.println(Arrays.toString(k.clusters));
        System.out.println(randIndex(k.assignments, inst));
    }

    //Class for calculating Shape Based Distance
    private class SBD {

        public double dist;
        public Instance yShift;

        private FFT fft;

        public SBD(){}

        public SBD(Instance first, Instance second, boolean calcShift){
            calculateDistance(first, second, calcShift);
        }

        public double calculateDistance(Instance first, Instance second){
            calculateDistance(first, second, false);
            return dist;
        }

        public void calculateDistance(Instance first, Instance second, boolean calcShift){
            int oldLength = first.numAttributes()-1;
            int oldLengthY = second.numAttributes()-1;

            int length = paddedLength(oldLength);

            //FFT and IFFT
            fft = new FFT();

            Complex[] firstC = fft(first, oldLength, length);
            Complex[] secondC = fft(second, oldLengthY, length);

            for (int i = 0; i < length; i++){
                secondC[i].conjugate();
                firstC[i].multiply(secondC[i]);
            }

            fft.inverseFFT(firstC, length);

            //Calculate NCCc values
            double firstNorm = sumSquare(first);
            double secondNorm = sumSquare(second);
            double norm = Math.sqrt(firstNorm * secondNorm);

            double[] ncc = new double[oldLength*2-1];
            int idx = 0;

            for (int i = length-oldLength+1; i < length; i++){
                ncc[idx++] = firstC[i].getReal()/norm;
            }

            for (int i = 0; i < oldLength; i++){
                ncc[idx++] = firstC[i].getReal()/norm;
            }

            double maxValue = 0;
            int shift = -1;

            //Largest NCCc value and index
            for (int i = 0; i < ncc.length; i++){
                if (ncc[i] > maxValue){
                    maxValue = ncc[i];
                    shift = i;
                }
            }

            dist = 1 - maxValue;

            //Create y', shifting the second instance in a direction and padding with 0s
            if (calcShift){
                if (oldLength > oldLengthY){
                    shift -= oldLength-1;
                }
                else {
                    shift -= oldLengthY-1;
                }

                yShift = new DenseInstance(1, new double[second.numAttributes()]);

                if (shift >= 0){
                    for (int i = 0; i < oldLengthY-shift; i++){
                        yShift.setValue(i + shift, second.value(i));
                    }
                }
                else {
                    for (int i = 0; i < oldLengthY+shift; i++){
                        yShift.setValue(i, second.value(i-shift));
                    }
                }
            }
        }

        //Amount of padding required for FFT
        private int paddedLength(int oldLength){
            int length = (int)MathsPower2.roundPow2((float)oldLength);
            if (length < oldLength) length *= 2;
            return length;
        }

        //Run FFT and return array of complex numbers
        private Complex[] fft(Instance inst, int oldLength, int length){
            Complex[] complex = new Complex[length];

            for (int i = 0; i < oldLength; i++){
                complex[i] = new Complex(inst.value(i), 0);
            }

            for (int i = oldLength; i < length; i++){
                complex[i] = new Complex(0,0);
            }

            fft.fft(complex, length);

            return complex;
        }

        private double sumSquare(Instance inst){
            double sum = 0;

            for (int i = 0; i < inst.numAttributes(); i++){
                sum += inst.value(i)*inst.value(i);
            }

            return sum;
        }
    }
}
