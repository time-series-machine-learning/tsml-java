package tsml.classifiers.multivariate;

//import tsml.classifiers.distance_based.distances.old.DTW_D;
import utilities.generic_storage.Pair;
import utilities.multivariate_tools.MultivariateInstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import static utilities.InstanceTools.findMinDistance;


/**
 *
 * @author Alejandro Pasos Ruiz
 */
public abstract class MultivariateAbstractClassifier extends AbstractClassifier {

    public MultivariateAbstractClassifier(){
        super();
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.enable(Capabilities.Capability.RELATIONAL_ATTRIBUTES);
        result.disable(Capabilities.Capability.MISSING_VALUES);
        return result;
    }

    protected void testWithFailRelationalInstances(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
        for (Instance instance: data){
            testWithFailRelationalInstance(instance);
        }

    }

    protected void testWithFailRelationalInstance(Instance data) throws Exception {
            Instances group = MultivariateInstanceTools.splitMultivariateInstanceOnInstances(data);
            getCapabilities().testWithFail(group);
    }




}
