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
 
package tsml.classifiers.distance_based.utils.collections.params.distribution.int_based;

import org.junit.Assert;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class UniformIntDistribution extends IntDistribution {

    public UniformIntDistribution(final Integer max) {
        this(0, max);
    }

    public UniformIntDistribution(final Integer min, final Integer max) {
        super(min, max);
    }

    @Override
    protected void checkSetMinAndMax(final Integer min, final Integer max) {
        Assert.assertNotNull(min);
        Assert.assertNotNull(max);
        Assert.assertTrue(min <= max);
    }

    @Override
    public Integer sample() {
        int max = getMax();
        int min = getMin();
        return getRandom().nextInt(max - min + 1) + min;
    }
}
