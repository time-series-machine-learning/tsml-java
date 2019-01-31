    /*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.clusterers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import timeseriesweka.clusterers.AbstractTimeSeriesClusterer;
import weka.core.Instance;
import weka.core.Instances;
import timeseriesweka.filters.FFT;
import timeseriesweka.filters.FFT.Complex;
import static timeseriesweka.filters.FFT.MathsPower2;
import utilities.ClassifierTools;
import static utilities.Utilities.extractTimeSeries;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.matrix.EigenvalueDecomposition;
import weka.core.matrix.Matrix;

/**
 *
 * @author pfm15hbu
 */
public class KShape extends AbstractTimeSeriesClusterer {
    
    private int k = 2;
    int seed = Integer.MIN_VALUE;
    
    Instances centroids;
    double[] cluster;
    
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
        if (!changeOriginalInstances){
            data = new Instances(data);
        }
        
        zNormalise(data);
        int numAtts = data.numAttributes();
        
        if (hasClassValue){
            numAtts--;
        }
        
        ArrayList<Attribute> atts = new ArrayList(numAtts);
        
        for (int i = 0; i < numAtts; i++){
            atts.add(new Attribute("att" + i));
        }
        
        centroids = new Instances("centroids", atts, k);
        
        for (int i = 0; i < k; i++){
            centroids.add(new DenseInstance(numAtts));
        }
        
        System.out.println(centroids);
        
        Random rand;
        
        if (seed == Integer.MIN_VALUE){
            rand = new Random();
        }
        else{
            rand = new Random(seed);
        }
        
        int iterations = 0;
        cluster = new double[data.numInstances()];
        
        for (int i = 0; i < cluster.length; i++){
            cluster[i] = Math.round(rand.nextDouble()*k);
        }
        
        double[] prevCluster = new double[data.numInstances()];
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
                    
                    if (sbd.dist < minDist){
                        minDist = sbd.dist;
                        cluster[i] = n;
                    }
                }
            }
            
            iterations++;
        }
    }
    
    private Instance shapeExtraction(Instances data, Instance centroid, int centroidNum){
        Instances subsample = new Instances(data, 0);
        int seriesSize = centroid.numAttributes();
        double sum = 0;
        
        for (int i = 0; i < seriesSize; i++){
            if (i != centroid.classIndex()){
                sum += centroid.value(i);
            }
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
            return new DenseInstance(seriesSize);
        }
        
        double[][] subsampleArray = new double[subsample.numInstances()][];
        
        for (int i = 0; i < subsample.numInstances(); i++){
            System.out.println(subsample.get(i));
            subsampleArray[i] = extractTimeSeries(subsample.get(i));
        }
        
        Matrix matrix = new Matrix(subsampleArray);
        Matrix matrixT = matrix.transpose();
        matrix = matrix.times(matrixT);
        
        Matrix identity = Matrix.identity(seriesSize, seriesSize);
        Matrix ones = new Matrix(seriesSize, seriesSize, 1);
        ones = ones.times(1.0/seriesSize);
        identity.minus(ones);
        
        matrix = identity.times(matrix).times(identity);
        
        EigenvalueDecomposition eig = new EigenvalueDecomposition(matrix);
        double[] eigVector = eig.getV().getArray()[0];
        
        return new DenseInstance(1, eigVector);
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

        Instances inst = ClassifierTools.loadData("Z:/Data/TSCProblems2018/Adiac/ADIAC_TRAIN.arff");
        inst.setClassIndex(inst.numAttributes()-1);
        KShape k = new KShape();
        k.seed = 1;
        k.buildClusterer(inst);
        System.out.println(Arrays.toString(k.cluster));
        System.out.println(k.centroids);
    }
    
    private class SBD{
        
        public double dist;
        public Instance yShift;
        
        private FFT fft;
    
        public SBD(Instance first, Instance second, boolean calcShift){
            calculateDistance(first, second, calcShift);
        }

        private void calculateDistance(Instance first, Instance second, boolean calcShift){
            int oldLength;
            int oldLengthY;
            
            if(hasClassValue){
                oldLength = first.numAttributes()-1;
                oldLengthY = second.numAttributes()-1;

            }
            else{
                oldLength = first.numAttributes();
                oldLengthY = second.numAttributes();
            }
            
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

                yShift = new DenseInstance(oldLengthY);

                if (shift >= 0){
                    for (int i = 0; i < second.numAttributes()-shift; i++){
                        yShift.setValue(i + shift, second.value(i));
                    }
                }
                else {
                    for (int i = 0; i < second.numAttributes()+shift; i++){
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
