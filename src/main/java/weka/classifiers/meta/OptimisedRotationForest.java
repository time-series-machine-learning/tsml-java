/*
Wrapper for rotation forest that peforms a Naive model selection

Based on findings from
http://pages.bangor.ac.uk/~mas00a/papers/lkjrmcs07.pdf

Search for

Number of Feature Subsets, K=
Number of features in a subset: M = 3;
Number of classifiers in the ensemble, L=

Maybe try get the OOB errpr?
the feature set is split randomly into K subsets, principal component analysis (PCA) is
run separately on each subset, and a new set of n linear extracted features
is constructed by pooling all principal components.
 */
package weka.classifiers.meta;

import utilities.ClassifierTools;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class OptimisedRotationForest extends RotationForest{
    static int[] MVALUES={1,2,3,4,5,6,7,8,9,10};
    static int[] LVALUES={5,10,15,20,25,30,40,50,60,70};
    int folds=10;
    
    @Override
    public void buildClassifier(Instances train) throws Exception{
        int bestM=0;
        double bestAcc=0;
        int bestL=0;
        if(train.numInstances()<folds)
            folds=train.numInstances();
        for(int m:MVALUES){
            for( int l:LVALUES){
                RotationForest trainer=new RotationForest();
                trainer.setMaxGroup(m);
                trainer.setMinGroup(m);
                trainer.setNumIterations(l);
                double acc=ClassifierTools.stratifiedCrossValidation(train, trainer, folds, 0);
                if(acc>bestAcc){
                    bestM=m;
                    bestL=l;
                    bestAcc=acc;
                }
            }
        }
        setMaxGroup(bestM);
        setMinGroup(bestM);
        setNumIterations(bestL);
        super.buildClassifier(train);
    }
}
