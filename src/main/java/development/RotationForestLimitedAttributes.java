/*
 Adjusted Rotation Forest.

VERSION 1: 

1. Limit the max number of attributes per tree
    Test 1: make sure it still does the same thing when maxNumAttributes> numAtts in all cases
10/2/17: Run TunedRotationForest and RotationForestLimitedAttributes with 
maxNumAttributes=10000 
should be no difference: 

    Test 2: check it still runs with problems where maxNumAttributes> numAtts
    Test 3: Compare accuracy on problems where maxNumAttributes> numAtts
    Test 4: Perform timing experiment on problems where maxNumAttributes> numAtts

Timing Experiment: 
Decide on threshold. 

1. Determine problems that take more than 1 day or 1 hour to train a single model on my new machine
2. Generate times for a range of n and m.
3. Construct linear model as a function of n and m


Version 2 will. Impose bagging and work out OOB Error
 */
package development;

import java.util.Random;
import weka.classifiers.meta.RotationForest;
import vector_classifiers.TunedRotationForest;
import vector_classifiers.TunedRotationForest;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class RotationForestLimitedAttributes extends TunedRotationForest{
    private int maxNumAttributes=100;
    private boolean propToNumAtts=true;
    public RotationForestLimitedAttributes(){
        this.estimateAccFromTrain(false);
        this.tuneParameters(false);
        this.setNumIterations(200);
    }
    public void setMaxNumAttributes(int m){
        if(m<100)
            maxNumAttributes=m;
        else 
           maxNumAttributes=(int)Math.sqrt((double)m);
    }
    @Override
    protected int [] attributesPermutation(int numAttributes, int classAttribute,
                                         Random random) {

    int [] permutation = new int[numAttributes-1];
    int i = 0;
//This just ignores the class attribute
    for(; i < classAttribute; i++){
      permutation[i] = i;
    }
    for(; i < permutation.length; i++){
      permutation[i] = i + 1;
    }

    permute( permutation, random );
    if(numAttributes>maxNumAttributes){
//TRUNCTATE THE PERMATION TO CONSIDER maxNumAttributes. 
// we could do this more efficiently, but this is the simplest way. 
        int[] temp = new int[maxNumAttributes];
       System.arraycopy(permutation, 0, temp, 0, maxNumAttributes);
       permutation=temp;
    }
    
    return permutation;
  }    
    @Override
    public void buildClassifier(Instances data) throws Exception{
       if (propToNumAtts)
        setMaxNumAttributes(data.numAttributes()-1);   
       super.buildClassifier(data);
    }
//Bagging    
    public static void main(String[] args){
        
    }
}
