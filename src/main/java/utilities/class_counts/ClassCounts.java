/*
 This is used by Aarons shapelet code and is down for depreciation
 */
package utilities.class_counts;

import java.io.Serializable;
import java.util.Collection;
import java.util.Set;

/**
 *
 * @author raj09hxu
 */
public abstract class ClassCounts implements Serializable {
    public abstract int get(double classValue);
    public abstract int get(int accessValue);
    public abstract void put(double classValue, int value);
    public abstract int size();
    public abstract Set<Double> keySet();
    public abstract Collection<Integer> values();
    public abstract void addTo(double classValue, int val);
}
