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
 
package tsml.classifiers.distance_based.utils.system.memory;

/**
 * Purpose: get stats related to memory.
 *
 * Contributors: goastler
 */
public interface MemoryWatchable {
    long getMaxMemoryUsage();
    
    
    static void gc() {
        // do it twice and this automagically cleans up memory somehow...
        System.gc(); 
        System.gc();
        // above may have put some objs in a queue for finalization, so let's clear them out
        System.runFinalization();
        System.runFinalization();
    }
}
