package tsml.transformers;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.PrincipalComponents;

/**
 * Filter to transform data into principle components. Currently just a wrapper for the Weka Filter PrincipalComponents
 *
 */
public class PCA extends PrincipalComponents implements Transformer {
//    PrincipalComponents pc=new weka.filters.unsupervised.attribute.PrincipalComponents();
    int numAttributesToKeep=100;
    @Override
    public void fit(Instances data) {
        if(data.numAttributes()-1<numAttributesToKeep)
            numAttributesToKeep=data.numAttributes()-1;
        setMaximumAttributes(numAttributesToKeep);

    }

    @Override
    public Instances transform(Instances data) {
        return null;
    }

    @Override
    public Instance transform(Instance inst) {
        return null;
    }

    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        if(data.numAttributes()-1<numAttributesToKeep)
            numAttributesToKeep=data.numAttributes()-1;
        setMaximumAttributes(numAttributesToKeep);
        try{
            return super.determineOutputFormat(data);
        }catch(Exception e){
            throw new IllegalArgumentException(e.getMessage());
        }
    }
}
