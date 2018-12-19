
package timeseriesweka.classifiers.ensembles.elastic_ensemble;

import java.util.Comparator;
import java.util.PriorityQueue;
import static utilities.GenericTools.indexOfMax;
import utilities.generic_storage.Pair;
import weka.core.Instance;

/**
 * Extension of DTW1NN to allow different values of k, originally written 
 * to include in HESCA (for timeseries data) so that DTW returns reasonable probability distributions, 
 * instead of a zero-one vector 
 * 
 * ***DO NOT USE 
 * 
 * There's some mega edge case where the distributionForInstance returns a distribution with NaN values, 
 * found only in ElectricDevices (so far) 99.9...% of test predictions were fine, enough for me personally 
 * to show that 1NN > 5NN > 11NN anyways, so I'm not going to bother spending the time fixing it 
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class DTWKNN extends DTW1NN {
    
    public int k; 
    
    private static Comparator<Pair<Double, Integer>> comparator = new Comparator<Pair<Double, Integer>>() { 
        @Override
        public int compare(Pair<Double, Integer> o1, Pair<Double, Integer> o2) {
            return o1.var1.compareTo(o2.var1) * (-1); //reverse ordering
        }
    };
    
    public DTWKNN() {
        super();
        k = 5;
    }

    public DTWKNN(int k) {
        super();
        this.k = k;
    }
    
    public DTWKNN(double r) {
        super(r);
        k = 5;
    }
    
    public DTWKNN(double r, int k) {
        super(r);
        this.k = k;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return indexOfMax(distributionForInstance(instance));
    }

    @Override
    public double[] distributionForInstance(Instance testInst) throws Exception {
        //the pair is <distance, classvalue> 
        PriorityQueue<Pair<Double, Integer>> topK = new PriorityQueue<>(k, comparator);
        
        double thisDist = distance(testInst, train.instance(0), Double.MAX_VALUE); 
        topK.add(new Pair<>(thisDist, (int)train.instance(0).classValue()));
                
        for(int i = 1; i < train.numInstances(); ++i){
            Instance trainInst = train.instance(i);
            thisDist = distance(testInst, trainInst, topK.peek().var1); 
            
            if (topK.size() < k) //not yet full
                topK.add(new Pair<>(thisDist, (int)trainInst.classValue()));
            else if(thisDist < topK.peek().var1){
                topK.poll();
                topK.add(new Pair<>(thisDist, (int)trainInst.classValue()));
            }
        }
        
        double distanceSum = .0;
        for (Pair<Double, Integer> pair : topK)
            distanceSum += pair.var1;
        
        double[] distribution = new double[train.numClasses()];
        
        //todo must be some way to do it in a single loop just brain farting on it right now and it's not important
//        for (Pair<Double, Integer> pair : topK) {
//            double voteWeight = (distanceSum - pair.var1) / distanceSum; 
//            distribution[pair.var2] += voteWeight;
//        }

        double distanceSum2 = .0;
        for (Pair<Double, Integer> pair : topK) {
            pair.var1 = 1 - (pair.var1 / distanceSum);
            distanceSum2 += pair.var1;
        }
        for (Pair<Double, Integer> pair : topK)
            distribution[pair.var2] += pair.var1 / distanceSum2;
       

        
        return distribution;
    }
}
