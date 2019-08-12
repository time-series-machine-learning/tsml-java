package timeseriesweka.classifiers.distance_based.ee.selection;

import utilities.Copyable;

import java.util.List;

public interface Selector<A>
    extends Copyable {
    void add(A candidate);
    List<A> getSelected();
    void clear();
}
