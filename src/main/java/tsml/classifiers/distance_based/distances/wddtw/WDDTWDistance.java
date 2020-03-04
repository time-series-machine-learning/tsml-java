package tsml.classifiers.distance_based.distances.wddtw;

import tsml.classifiers.distance_based.distances.transformed.TransformedDistanceMeasure;
import tsml.classifiers.distance_based.distances.wdtw.WDTW;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistance;
import tsml.filters.CachedFilter;
import tsml.filters.Derivative;


/**
 * WDDTW distance measure.
 * <p>
 * Contributors: goastler
 */
public class WDDTWDistance extends TransformedDistanceMeasure implements WDTW {

    // Global derivative function which is cached, i.e. if you ask it to convert the same instance twice it will
    // instead fetch from the cache the second time
    private CachedFilter derivativeCache;
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
}
