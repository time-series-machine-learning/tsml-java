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
 
package tsml.classifiers.distance_based.utils.collections.box;

import java.util.function.Supplier;

/**
 * Purpose: defer an operation until needed.
 * <p>
 * Contributors: goastler
 */
public class DeferredBox<A> {

    private final Supplier<A> getter;
    private A object;

    public DeferredBox(final Supplier<A> getter) {
        this.getter = getter;
    }

    public A get() {
        if(object == null) {
            object = getter.get();
        }
        return object;
    }

}
