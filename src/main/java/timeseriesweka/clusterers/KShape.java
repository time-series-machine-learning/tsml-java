    /*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.clusterers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import timeseriesweka.filters.FFT;
import timeseriesweka.filters.FFT.Complex;
import static timeseriesweka.filters.FFT.MathsPower2;
import utilities.ClassifierTools;

import static utilities.ClusteringUtilities.randIndex;
import static utilities.ClusteringUtilities.zNormalise;
import static utilities.InstanceTools.deleteClassAttribute;
import static utilities.Utilities.extractTimeSeries;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.matrix.EigenvalueDecomposition;
import weka.core.matrix.Matrix;

/**
 * UNFINISHED AND UNTESTED, DONT USE.
 *
 * @author pfm15hbu
 */
public class KShape extends AbstractTimeSeriesClusterer {
    
    private int k = 2;
    private int seed = Integer.MIN_VALUE;
    
    private Instances centroids;
    
    public KShape(){}
    
    @Override
    public int numberOfClusters(){
        return k;
    }
    
    public void setK(int k){
        this.k = k;
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        if (!dontCopyInstances){
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
        cluster = new int[data.numInstances()];
        
        for (int i = 0; i < cluster.length; i++){
            cluster[i] = (int)Math.ceil(rand.nextDouble()*k)-1;
        }

        int[] prevCluster = new int[data.numInstances()];
        prevCluster[0] = -1;
        
        while (!Arrays.equals(cluster, prevCluster) && iterations < 100){
            prevCluster = Arrays.copyOf(cluster, cluster.length);

            for (int i = 0; i < k; i ++){
                centroids.set(i, shapeExtraction(data, centroids.get(i), i));
            }

            for (int i = 0; i < data.numInstances(); i++){
                double minDist = Double.MAX_VALUE;
                
                for (int n = 0; n < k; n++){
                    SBD sbd = new SBD(centroids.get(n), data.get(i), false);
                    //System.out.println(sbd.dist);
                    if (sbd.dist < minDist){
                        minDist = sbd.dist;
                        cluster[i] = n;
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
                if(n == cluster[i]){
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
        
        for (int i = 0; i < data.numInstances(); i++){
            if (cluster[i] == centroidNum){
                if (sumZero){
                    subsample.add(data.get(i));
                }
                else{
                    SBD sbd = new SBD(centroid, data.get(i), true);
                    subsample.add(sbd.yShift);
                }
            }
        }

        if (subsample.numInstances() == 0){
            return new DenseInstance(1, new double[centroid.numAttributes()]);
        }

        zNormalise(subsample);

        double[][] subsampleArray = new double[subsample.numInstances()][];

        for (int i = 0; i < subsample.numInstances(); i++){
            subsampleArray[i] = subsample.get(i).toDoubleArray();
        }
        
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

            if (Math.round(eigSum) == subsample.get(0).numAttributes() && Math.round(eigSumNeg) == subsample.get(0).numAttributes()){
                col++;
                System.err.println("Possible eig error");
            }
            else{
                break;
            }
        }

        Instance newCent;

        if (Math.sqrt(eigSum) < Math.sqrt(eigSumNeg)){
            newCent = new DenseInstance(1, eigVector);
        }
        else{
            newCent = new DenseInstance(1, eigVectorNeg);
        }

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
        Instances inst = ClassifierTools.loadData("D:\\CMP Machine Learning\\Datasets\\TSC Archive\\" + dataset + "/" + dataset + "_TRAIN.arff");
        Instances inst2 = ClassifierTools.loadData("D:\\CMP Machine Learning\\Datasets\\TSC Archive\\" + dataset + "/" + dataset + "_TEST.arff");
        inst.setClassIndex(inst.numAttributes()-1);
        inst.addAll(inst2);

        KShape k = new KShape();
        k.seed = 0;
        k.k = inst.numClasses();
        k.buildClusterer(inst);

        System.out.println(k.clusters.length);
        System.out.println(Arrays.toString(k.clusters));
        System.out.println(randIndex(k.cluster, inst));
    }
    
    private class SBD{
        
        public double dist;
        public Instance yShift;
        
        private FFT fft;
    
        public SBD(Instance first, Instance second, boolean calcShift){
            calculateDistance(first, second, calcShift);
        }

        private void calculateDistance(Instance first, Instance second, boolean calcShift){
            int oldLength = first.numAttributes();
            int oldLengthY = second.numAttributes();

            int length = paddedLength(2*oldLength-1);
            
            fft = new FFT();
                    
            Complex[] firstC = fft(first, oldLength, length);
            Complex[] secondC = fft(second, oldLengthY, length);
            
            for (int i = 0; i < length; i++){
                secondC[i].conjugate();
                firstC[i].multiply(secondC[i]);
            }
            
            fft.inverseFFT(firstC, length);
            
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
                    
            for (int i = 0; i < ncc.length; i++){
                if (ncc[i] > maxValue){
                    maxValue = ncc[i];
                    shift = i;
                }
            }
            
            dist = 1 - maxValue;
            
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
        
        private int paddedLength(int oldLength){
            int length;
            
            if(!MathsPower2.isPow2(oldLength)){
                length = (int)MathsPower2.roundPow2((float)oldLength);
                
                if(length<oldLength){
                    length *= 2;
                }
            }
            else{
                length = oldLength;
            }
            
            return length;
        }
        
        public Complex[] fft(Instance inst, int oldLength, int length){
            Complex[] complex = new Complex[length];
            
            for (int i = 0; i < oldLength; i++){
                complex[i] = new Complex(inst.value(i), 0);
            }
            
            while(oldLength < length){
                complex[oldLength++] = new Complex(0,0);
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
