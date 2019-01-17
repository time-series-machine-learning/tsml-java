package timeseriesweka.clusterers;

import java.util.ArrayList;
import java.util.Arrays;

import vector_clusterers.KMeans;
import weka.clusterers.AbstractClusterer;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author pfm15hbu
 */
public class UnsupervisedShapelets extends AbstractClusterer{
    
    int k;
    int seed = Integer.MIN_VALUE;
    
    ArrayList<UShapelet> shapelets;
    int numInstances;
    
    private int[] cluster;
    private ArrayList<Integer>[] clusters;
    
    int numFolds = 5;
    
    @Override
    public void buildClusterer(Instances data) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int numberOfClusters() throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    private void extractUShapelets(Instances data){
        int[] shapeletLengths = {10, 100, 1000};
        
        shapelets = new ArrayList();
        numInstances = data.size();
        //random?
        Instance inst = data.firstInstance();
        boolean finished = false;
        
        while (!finished){
            ArrayList<UShapelet> shapeletCandidates = new ArrayList();
            
            for (int i = 0; i < shapeletLengths.length; i++){
                for (int n = 0; n < inst.numAttributes() - shapeletLengths[i] /*+ 1*/; n++){
                    UShapelet candidate = new UShapelet(n, shapeletLengths[i], 0, inst);
                    candidate.computeGap(data);
                    shapeletCandidates.add(candidate);
                }
            }
            
            double maxGap = 0;
            int maxGapIndex = -1;
            
            for (int i = 0; i < shapeletCandidates.size(); i++){
                if (shapeletCandidates.get(i).gap > maxGap){
                    maxGap = shapeletCandidates.get(i).gap;
                    maxGapIndex = i;
                }
            }
            
            UShapelet best = shapeletCandidates.get(maxGapIndex);
            
            shapelets.add(best);
            
            double[] distances = best.computeDistances(data);
            ArrayList<Double> lesserDists = new ArrayList();
            double maxDist = 0;
            int maxDistIndex = -1;
            
            for (int i = 0; i < distances.length; i++){
                if (distances[i] < best.dt){
                    lesserDists.add(distances[i]);
                }
                else if (distances[i] > maxDist){
                    maxDist = distances[i];
                    maxDistIndex = i;
                }
            }
            
            if (lesserDists.size() == 1){
                finished = true;
            }
            else{
                inst = data.get(maxDistIndex);
                
                double mean = mean(lesserDists);
                double cutoff = mean + standardDeviation(lesserDists, mean);
                
                Instances newData = new Instances(data, 0);
                
                for (int i = 0; i < numInstances; i++){
                    if (distances[i] >= cutoff){
                        newData.add(data.get(i));
                    }
                }
                
                data = newData;
            }
        }
    }
    
    private void clusterData() throws Exception{
        
        ArrayList<Attribute> atts = new ArrayList();
        
        for (int i = 0; i < numInstances; i++){
            atts.add(new Attribute("att" + i));
        }
        
        Instances distanceMap = new Instances("temp", atts, 0);
        
        int[][] foldClusters = new int[shapelets.size()][];
        double minRandIndex = 1;
        int minIndex = -1;
        
        for (int i = 0; i < shapelets.size(); i++){
            UShapelet shapelet = shapelets.get(i);
            double[] distances = shapelet.distances;
            double minDist = Double.MAX_VALUE;
            
            distanceMap.add(new DenseInstance(1,distances));
            
            for (int n = 0; n < numFolds; n++){
                KMeans kmeans = new KMeans();
                kmeans.setK(k);
                kmeans.setNormaliseData(false);
                kmeans.setFindBestK(false);
                kmeans.setRefinedInitialMedoids(false);
                kmeans.setSeed(seed);
                kmeans.buildClusterer(distanceMap);
                
                double dist = kmeans.clusterSquaredDistance(distanceMap);
                
                if (dist < minDist){
                    minDist = dist;
                    foldClusters[i] = kmeans.getCluster();
                }
            }
            
            double randIndex = randIndex(foldClusters[i-1],foldClusters[i]);
            
            if (randIndex < minRandIndex){
                minRandIndex = randIndex;
                minIndex = i;
            }
        }
        
        cluster = foldClusters[minIndex];
        
        clusters = new ArrayList[k];
        
        for (int i = 0; i < k; i++){
            clusters[i] = new ArrayList();
        }
        
        for (int i = 0; i < numInstances; i++){
            for (int n = 0; n < k; n++){
                if(n == cluster[i]){
                    clusters[i].add(n);
                    break;
                }
            }
        }
    }
    
    private double randIndex(int[] clusters1, int[] clusters2){
        int samePairs = 0;
        int numPairs = 0;
        
        for (int i = 0; i < clusters1.length; i++){
            for (int n = 0; n < i; n++){
                if ((clusters1[i] == clusters1[n] && clusters2[i] == clusters2[n]) ||
                       (clusters1[i] != clusters1[n] && clusters2[i] != clusters2[n])){
                    samePairs++;
                }
            }
            
            numPairs += i;
        }
        
        return samePairs/numPairs;
    }
    
    private double mean(ArrayList<Double> dists){
        double meanSum = 0;
                
        for (int i = 0; i < dists.size(); i++){
            meanSum += dists.get(i);
        }
                
        return meanSum /= dists.size();
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
        
    }
    
    private class UShapelet{
    
        public int startPoint;
        public int length;
        public int seriesIndex;
        public double[] series;
        
        public double gap;
        public double dt;
        
        public double[] distances;
        
        public UShapelet(int startPoint, int length, int seriesIndex, Instance inst){
            this.startPoint = startPoint;
            this.length = length;
            this.seriesIndex = seriesIndex;
            this.series = inst.toDoubleArray();
        }
        
        public UShapelet(int startPoint, int length, Instance inst){
            this.startPoint = startPoint;
            this.length = length;
            this.series = inst.toDoubleArray();
        }
        
        public void computeGap(Instances data){
            this.distances = computeDistances(data);
            
            double[] sortedDistances = Arrays.copyOf(distances, distances.length);
            Arrays.sort(sortedDistances);

            for (int i = 0; i < sortedDistances.length-1; i++){
                double dist = (sortedDistances[i] + sortedDistances[i+1])/2;

                ArrayList<Double> lesserDists = new ArrayList();
                ArrayList<Double> greaterDists = new ArrayList();

                for (int n = 0; n < sortedDistances.length; n++){
                    if (sortedDistances[n] < dist){
                        lesserDists.add(sortedDistances[n]);
                    }
                    else{
                        greaterDists.add(sortedDistances[n]);
                    }
                }

                double ratio = lesserDists.size()/greaterDists.size();

                if (1/k < ratio){
                    double lesserMean = mean(lesserDists);
                    double greaterMean = mean(greaterDists);
                    double lesserStdev = standardDeviation(lesserDists, lesserMean);
                    double greaterStdev = standardDeviation(greaterDists, greaterMean);

                    double gap = greaterMean - greaterStdev - (lesserMean + lesserStdev);

                    if (gap > this.gap){
                        this.gap = gap;
                        this.dt = dist;
                    }
                }
            }
        }
        
        public double[] computeDistances(Instances data){
            
            if (distances != null){
                return distances;
            }
            
            double[] distances = new double[numInstances];
            double[] shapelet = zNormalise();
            
            for (int i = 0; i < numInstances; i++){
                Instance inst = data.get(i);
                distances[i] = Double.MAX_VALUE;
                UShapelet subseries = new UShapelet(0, length, inst);

                for (int n = 0; n < inst.numAttributes() - length /*+ 1*/; n++){
                    subseries.length = n;
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
        
        private double[] zNormalise(){
            double meanSum = 0;

            for (int i = startPoint; i < startPoint + length; i++){
                meanSum += series[i];
            }

            double mean = meanSum /= length;

            double stdevSum = 0;
            double temp;

            for (int i = startPoint; i < startPoint + length; i++){
                temp = series[i] - mean;
                stdevSum += temp * temp;
            }

            double meanOfDiffs = stdevSum/length;
            double stdev = Math.sqrt(meanOfDiffs);

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
