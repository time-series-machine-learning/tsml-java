/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package utilities;

import com.beust.jcommander.internal.Lists;
import java.math.BigDecimal;
import java.math.RoundingMode;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import tsml.classifiers.distance_based.utils.collections.views.IntListView;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializedObject;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.regex.Pattern;

public class Utilities {

    public static <A> int sum(Iterator<A> iterator, Function<A, Integer> func) {
        int sum = 0;
        while(iterator.hasNext()) {
            A next = iterator.next();
            Integer integer = func.apply(next);
            sum += integer;
        }
        return sum;
    }

    public static <A, B> List<B> convert(Iterable<A> source, Function<A, B> converter) { // todo stream version
        return convert(source.iterator(), converter);
    }

    public static <A, B> List<B> convert(Iterator<A> source, Function<A, B> converter) {
        return convert(source, converter, ArrayList::new);
    }

    public static <A, B, C extends Collection<B>> C convert(Iterator<A> source, Function<A, B> converter, Supplier<C> supplier) {
        C destination = supplier.get();
        while(source.hasNext()) {
            A item = source.next();
            B convertedItem = converter.apply(item);
            destination.add(convertedItem);
        }
        return destination;
    }

    public static <A, B, C extends Collection<B>> C convert(Iterable<A> source, Function<A, B> converter, Supplier<C> supplier) {
        return convert(source.iterator(), converter, supplier);
    }

/**

* 6/2/19: bug fixed so it properly ignores the class value, only place its used
* is in measures.DistanceMeasure
 * @param instance
 * @return array of doubles with the class value removed
*/
    public static final double[] extractTimeSeries(Instance instance) {
        if(instance.classIndex() < 0) {
            return instance.toDoubleArray();
        } else {
            double[] timeSeries = new double[instance.numAttributes() - 1];
            for(int i = 0; i < instance.numAttributes(); i++) {
                if(i < instance.classIndex()) {
                    timeSeries[i] = instance.value(i);
                } else if (i != instance.classIndex()){
                    timeSeries[i - 1] = instance.value(i);
                }
            }
            return timeSeries;
        }
    }

    public static final int maxIndex(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if(array[maxIndex] < array[i]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static final int minIndex(double[] array) {
        int minIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if(array[i] < array[minIndex]) {
                minIndex = i;
            }
        }
        return minIndex;
    }

    public static double log(double value, double base) { // beware, this is inaccurate due to floating point error!
        if(value == 0) {
            return 0;
        }
        return Math.log(value) / Math.log(base);
    }

    /**
     * get the instances by class. This returns a map of class value to indices of instances in that class.
     * @param instances
     * @return
     */
    public static Map<Double, List<Integer>> instancesByClass(Instances instances) {
        Map<Double, List<Integer>> map = new LinkedHashMap<>(instances.size(), 1);
        for(int i = 0; i < instances.size(); i++) {
            final Instance instance = instances.get(i);
            map.computeIfAbsent(instance.classValue(), k -> new ArrayList<>()).add(i);
        }
        return map;
    }

    public static <A, B> boolean isUnique(final Iterator<A> iterator, Function<A, B> func) {
        if(!iterator.hasNext()) {
            return true;
        }
        B value = func.apply(iterator.next());
        while(iterator.hasNext()) {
            B nextValue = func.apply(iterator.next());
            if(value == null) {
                if(nextValue != null) {
                    return false;
                }
            } else if(!value.equals(nextValue)) {
                return false;
            }
        }
        return true;
    }

    public static <A> boolean isUnique(final Iterator<A> iterator) {
        return isUnique(iterator, i -> i);
    }

    public static <A> boolean isUnique(Iterable<A> iterable) {
        return isUnique(iterable.iterator());
    }

    public static <A, B> boolean isUnique(Iterable<A> iterable, Function<A, B> func) {
        return isUnique(iterable.iterator(), func);
    }

    public static boolean isHomogeneous(List<Instance> data) {
        return isUnique(data, Instance::classValue);
    }

    public static int[] argMax(double[] array) {
        List<Integer> indices = new ArrayList<>();
        double max = array[0];
        indices.add(0);
        for(int i = 1; i < array.length; i++) {
            if(array[i] >= max) {
                if(array[i] > max) {
                    max = array[i];
                    indices.clear();
                }
                indices.add(i);
            }
        }
        int[] indicesCopy = new int[indices.size()];
        for(int i = 0; i < indicesCopy.length; i++) {
            indicesCopy[i] = indices.get(i);
        }
        return indicesCopy;
    }

    public static int argMax(double[] array, Random random) {
        int[] indices = argMax(array);
        if(indices.length == 1) {
            return indices[0];
        }
        return indices[random.nextInt(indices.length)];
    }
   
    public static boolean stringIsDouble(String input){
         /*********Aarons Nasty Regex From https://docs.oracle.com/javase/8/docs/api/java/lang/Double.html#valueOf-java.lang.String **********/
        final String Digits     = "(\\p{Digit}+)";
        final String HexDigits  = "(\\p{XDigit}+)";
        // an exponent is 'e' or 'E' followed by an optionally
        // signed decimal integer.
        final String Exp        = "[eE][+-]?"+Digits;
        final String fpRegex    =
            ("[\\x00-\\x20]*"+  // Optional leading "whitespace"
            "[+-]?(" + // Optional sign character
            "NaN|" +           // "NaN" string
            "Infinity|" +      // "Infinity" string
    
            // A decimal floating-point string representing a finite positive
            // number without a leading sign has at most five basic pieces:
            // Digits . Digits ExponentPart FloatTypeSuffix
            //
            // Since this method allows integer-only strings as input
            // in addition to strings of floating-point literals, the
            // two sub-patterns below are simplifications of the grammar
            // productions from section 3.10.2 of
            // The Java Language Specification.
    
            // Digits ._opt Digits_opt ExponentPart_opt FloatTypeSuffix_opt
            "((("+Digits+"(\\.)?("+Digits+"?)("+Exp+")?)|"+
    
            // . Digits ExponentPart_opt FloatTypeSuffix_opt
            "(\\.("+Digits+")("+Exp+")?)|"+
    
            // Hexadecimal strings
            "((" +
            // 0[xX] HexDigits ._opt BinaryExponent FloatTypeSuffix_opt
            "(0[xX]" + HexDigits + "(\\.)?)|" +
    
            // 0[xX] HexDigits_opt . HexDigits BinaryExponent FloatTypeSuffix_opt
            "(0[xX]" + HexDigits + "?(\\.)" + HexDigits + ")" +
    
            ")[pP][+-]?" + Digits + "))" +
            "[fFdD]?))" +
            "[\\x00-\\x20]*");// Optional trailing "whitespace"

        return Pattern.matches(fpRegex, input);
    }

}
