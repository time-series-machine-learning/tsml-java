/*
 Base class to create folds from a dataset in a reproducable way. 
This class can be subtyped to allow for dataset specific cross validation.
Examples include leave one person out (e.g. EpilepsyX) or leave one bottle out
(e.g. EthanolLevel)

 */
package utilities;

import utilities.InstanceTools;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class FoldCreator {
    double prop=0.3;
    protected boolean deleteFirstAttribute=false;//Remove an index
    public void deleteFirstAtt(boolean b){
        deleteFirstAttribute=b;
    }
    public FoldCreator(){
        
    }
    public FoldCreator(double p){
        prop=p;
    }
    public void setProp(double p){
        prop=p;
    }
    public Instances[] createSplit(Instances data, int fold) throws Exception{
        Instances[] split= InstanceTools.resampleInstances(data, fold, prop);
        if(deleteFirstAttribute){
            split[0].deleteAttributeAt(0);
            split[1].deleteAttributeAt(0);
        }
        return split;
    } 
    
}
