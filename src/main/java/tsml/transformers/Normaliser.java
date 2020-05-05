package tsml.transformers;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Transformer to normalise time series. It can either standardise (remove series mean) or normalise (remove mean, unit variance)
 *
 *  * todo: implement univariate
 *  * todo: implement multivariate
 *  * todo: test
 * @author Tony Bagnall 18/4/2020
 */

public class Normaliser implements Transformer{
    @Override
    public void fit(Instances data) {

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
        return null;
    }
}
