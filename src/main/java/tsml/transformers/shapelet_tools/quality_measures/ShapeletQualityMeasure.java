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
package tsml.transformers.shapelet_tools.quality_measures;

import java.util.List;
import tsml.transformers.shapelet_tools.OrderLineObj;
import utilities.class_counts.ClassCounts;

/**
 *
 * @author raj09hxu
 */
    
    public interface ShapeletQualityMeasure 
    {
        public double calculateQuality(List<OrderLineObj> orderline, ClassCounts classDistribution);

        public double calculateSeperationGap(List<OrderLineObj> orderline);
    }
