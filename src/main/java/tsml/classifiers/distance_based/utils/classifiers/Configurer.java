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
 
package tsml.classifiers.distance_based.utils.classifiers;

import weka.classifiers.Classifier;

public interface Configurer<A> {

    <B extends A> B configure(B classifier);

    default Configurer<A> and(Configurer<A> other) {
        final Configurer<A> current = this;
        return new Configurer<A>() {
            @Override public <B extends A> B configure(B classifier) {
                classifier = other.configure(classifier);
                classifier = current.configure(classifier);
                return classifier;
            }
        };
    }
}
