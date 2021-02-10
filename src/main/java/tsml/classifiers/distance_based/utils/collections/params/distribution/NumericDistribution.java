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
 
package tsml.classifiers.distance_based.utils.collections.params.distribution;

import java.util.Objects;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public abstract class NumericDistribution<A extends Number> extends Distribution<A> {

    private A min;
    private A max;

    public NumericDistribution(final A min, final A max) {
        setMinAndMax(min, max);
    }

    protected abstract void checkSetMinAndMax(A min, A max);

    public void setMinAndMax(A min, A max) {
        checkSetMinAndMax(min, max);
        this.min = min;
        this.max = max;
    }

    public abstract A sample();

    public A getMax() {
        return max;
    }

    public void setMax(final A max) {
        setMinAndMax(min, max);
    }

    public A getMin() {
        return min;
    }

    public void setMin(final A min) {
        setMinAndMax(min, max);
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "{" +
            "min=" + min +
            ", max=" + max +
            '}';
    }

    @Override
    public boolean equals(final Object o) {
        if(this == o) {
            return true;
        }
        if(o == null || getClass() != o.getClass()) {
            return false;
        }
        final NumericDistribution<?> that = (NumericDistribution<?>) o;
        return Objects.equals(getMin(), that.getMin()) &&
            Objects.equals(getMax(), that.getMax());
    }

    @Override
    public int hashCode() {
        return Objects.hash(getMin(), getMax());
    }
}
