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

import akka.event.Logging;

import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Handle different log levels for providing anywhere from none to verbose output via logging.
 *
 * Contributors: goastler
 */
public interface Loggable {
    default Level getLogLevel() {
        return getLogger().getLevel();
    }
    
    default void setLogLevel(Level level) {
        getLogger().setLevel(level);
    }
    
    default void setLogLevel(String level) {
        setLogLevel(Level.parse(level.toUpperCase()));
    }
    
    Logger getLogger();

    /**
     * Manually specify the logger to log to. This is helpful to share loggers between classes / insts.
     * @param logger
     */
    void setLogger(Logger logger);
}
