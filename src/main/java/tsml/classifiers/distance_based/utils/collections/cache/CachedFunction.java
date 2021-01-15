/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.classifiers.distance_based.utils.collections.cache;

import java.io.Serializable;
import java.util.HashMap;
import java.util.function.Function;

/**
 * Purpose: class to cache the result of a function.
 * @param <I> the input type of the function.
 * @param <O> the output type of the function
 *
 * Contributors: goastler
 */
public class CachedFunction<I, O> implements Function<I, O>,
    Serializable {

    private final HashMap<I, O> cache = new HashMap<>();

    private Function<I, O> function;

    public CachedFunction(final Function<I, O> function) {
        this.function = function;
    }

    public O apply(I input) {
        return cache.computeIfAbsent(input, function);
    }

    public void clear() {
        cache.clear();
    }

    public Function<I, O> getFunction() {
        return function;
    }

    public void setFunction(Function<I, O> function) {
        if((function instanceof Serializable)) {
            function = (Serializable & Function<I, O>) function::apply;
        }
        this.function = function;
    }

}
