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
 
package tsml.classifiers.distance_based.utils.system.logging;

import java.util.logging.Logger;

/**
 * Purpose: simple interface to get the logger from an implementing class. Each logger is bespoke to an instance of a
 * class, therefore only access to that logger is granted, for the purpose of logging, and setting is not allowed
 * (otherwise we'd be logging from a logger unspecific to our context).
 *
 * Contributors: goastler
 */
public interface Loggable {
    Logger getLogger();

    default void setLogger(Logger logger) {

    }
}
