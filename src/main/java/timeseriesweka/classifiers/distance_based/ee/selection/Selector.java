package timeseriesweka.classifiers.distance_based.ee.selection;

import utilities.Copyable;

import java.util.List;
import java.util.Random;

public interface Selector<A>
    extends Copyable {
    boolean add(A candidate);
    List<A> getSelected();
    List<A> getSelectedWithDraws();
    void clear();
}
