package tsml.classifiers.distance_based.utils.collections;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public interface IndexedCollection<A> extends java.util.RandomAccess {

    A get(int index);

    int size();
}
