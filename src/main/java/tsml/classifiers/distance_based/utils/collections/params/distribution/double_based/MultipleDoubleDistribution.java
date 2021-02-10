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
 
package tsml.classifiers.distance_based.utils.collections.params.distribution.double_based;

import java.util.List;
import java.util.Random;

public class MultipleDoubleDistribution extends DoubleDistribution {

    private List<Double> sections;

    public MultipleDoubleDistribution(List<Double> sections) {
        super(0, 0);
        setSections(sections);
    }

    @Override public Double sample() {
        final Random random = getRandom();
        final int i = random.nextInt(sections.size() - 1);
        DoubleDistribution distribution = new UniformDoubleDistribution(sections.get(i), sections.get(i + 1));
        return distribution.sample(random);
    }

    public List<Double> getSections() {
        return sections;
    }

    public void setSections(final List<Double> sections) {
        this.sections = sections;
    }
}
