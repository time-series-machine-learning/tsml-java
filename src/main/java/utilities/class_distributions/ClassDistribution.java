/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities.class_distributions;

import java.io.Serializable;
import java.util.Collection;
import java.util.Set;

/**
 *
 * @author raj09hxu
 */
public abstract class ClassDistribution implements Serializable {
    public abstract int get(double classValue);
    public abstract int get(int accessValue);
    public abstract void put(double classValue, int value);
    public abstract int size();
    public abstract Set<Double> keySet();
    public abstract Collection<Integer> values();
    public abstract void addTo(double classValue, int val);
}
