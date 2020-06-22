package tsml.classifiers.distance_based.distances.wddtw;

import tsml.classifiers.distance_based.distances.transformed.TransformedDistanceMeasure;
import tsml.classifiers.distance_based.distances.wdtw.WDTW;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistance;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.transformers.CachedTransformer;
import tsml.transformers.Derivative;


/**
 * WDDTW distance measure.
 * <p>
 * Contributors: goastler
 */
public class WDDTWDistance extends TransformedDistanceMeasure implements WDTW {

    // Global derivative function which is cached, i.e. if you ask it to convert the same instance twice it will
    // instead fetch from the cache the second time
    private WDTW wdtw;

    public WDDTWDistance() {
        wdtw = new WDTWDistance();
        setTransformer(Derivative.getGlobalCache());
        setDistanceFunction(wdtw);
    }

    @Override
    public double getG() {
        return wdtw.getG();
    }

    @Override
    public void setG(double g) {
        wdtw.setG(g);
    }


    @Override public ParamSet getParams() {
        return wdtw.getParams(); // not including super params as we handle them manually in this class
    }

    @Override public void setParams(final ParamSet param) throws Exception {
        super.setParams(param);
        wdtw.setParams(param); // not including super params as we handle them manually in this class
    }
}
