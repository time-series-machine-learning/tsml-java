package tsml.classifiers.distance_based.utils.collections.iteration;

/**
 * Reset this class to uninitialised / default state.
 * <p>
 * Contributors: goastler
 */
public interface Resetable {
    void reset();
    
    static void reset(Object object) {
        if(object instanceof tsml.classifiers.distance_based.utils.collections.iteration.Resetable) {
            ((tsml.classifiers.distance_based.utils.collections.iteration.Resetable) object).reset();
        }
    }
}
