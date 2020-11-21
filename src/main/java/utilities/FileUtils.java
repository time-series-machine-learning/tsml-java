package utilities;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;

public class FileUtils {

    public static boolean isEmptyDir(String path) {
        final File file = new File(path);
        if(file.exists() && file.isDirectory()) {
            final File[] files = file.listFiles();
            return files == null || files.length == 0;
        }
        return true;
    }

    public static void writeToFile(String str, String path) throws
                                                            IOException {
        makeParentDir(path);
        BufferedWriter writer = new BufferedWriter(new FileWriter(path));
        writer.write(str);
        writer.close();
    }

    public static void writeToFile(String str, File file) throws
                                                          IOException {
        writeToFile(str, file.getPath());
    }

    public static String readFromFile(String path) throws
                                                   IOException {
        BufferedReader reader = new BufferedReader(new FileReader(path));
        String line;
        StringBuilder builder = new StringBuilder();
        while ((line = reader.readLine()) != null) {
            builder.append(line);
            builder.append(System.lineSeparator());
        }
        reader.close();
        return builder.toString();
    }

    public static String readFromFile(File path) throws
                                                 IOException {
        return readFromFile(path.getPath());
    }

    public static void copy(String src, String dest) throws IOException {
        copy(Paths.get(src), Paths.get(dest));
    }
    
    public static void copy(Path src, Path dest) throws IOException {
        try (Stream<Path> stream = Files.walk(src)) {
            stream.forEach(source -> {
                try {
                    Files.copy(source, dest.resolve(src.relativize(source)), REPLACE_EXISTING);
                } catch(IOException e) {
                    throw new IllegalStateException(e);
                }
            });
        }
    }

    public static void delete(String src) throws IOException {
        delete(Paths.get(src));
    }

    public static void delete(Path src) throws IOException {
        List<Path> paths = Files.walk(src).collect(Collectors.toList());
        Collections.reverse(paths);
        try (Stream<Path> stream = paths.stream()) {
            stream.forEach(source -> {
                try {
                    Files.delete(source);
                } catch(IOException e) {
                    throw new IllegalStateException(e);
                }
            });
        }
    }
    
    public static void makeDir(String filePath) {
        makeParentDir(new File(filePath));
    }
    
    public static void makeDir(File file) {
        makeParentDir(file);
        file.mkdirs();
    }

    public static void makeParentDir(String filePath) {
        makeParentDir(new File(filePath));
    }

    public static void makeParentDir(File file) {
        File parent = file.getParentFile();
        if(parent != null) {
            parent.mkdirs();
        }
    }

    public static boolean rename(final String src, final String dest) {
        makeParentDir(dest);
        return new File(src).renameTo(new File(dest));
    }

    public static class FileLock implements AutoCloseable {
        public static class LockException extends Exception {

            public LockException(String s) {
                super(s);
            }
        }

        private Thread thread;
        private File file;
        private File lockFile;
        private static final long interval = TimeUnit.MILLISECONDS.convert(1, TimeUnit.MINUTES);
        private static final long expiredInterval = interval + TimeUnit.MILLISECONDS.convert(1, TimeUnit.MINUTES);
        private final AtomicBoolean unlocked = new AtomicBoolean(true);

        public File getFile() {
            return file;
        }

        public File getLockFile() {
            return lockFile;
        }

        private FileLock(File file, boolean lock) throws LockException {
            this.file = file;
            this.lockFile = new File(file.getPath() + ".lock");
            if(lock) {
                lock();
            }
        }

        public FileLock(File file) throws LockException {
            // assume we want to lock the file asap
            this(file, true);
        }

        public FileLock(String path) throws LockException {
            this(new File(path));
        }

        private FileLock lock() throws LockException {
            if(isUnlocked()) {
                makeParentDir(lockFile);
                boolean stop = false;
                while(!stop) {
                    boolean created = false;
                    try {
                        created = lockFile.createNewFile();
                    } catch(IOException ignored) {}
                    if(created) {
                        lockFile.deleteOnExit();
                        unlocked.set(false);
                        thread = new Thread(this::watch);
                        thread.setDaemon(true);
                        thread.start();
                        return this;
                    } else {
                        long lastModified = lockFile.lastModified();
                        if(lastModified <= 0 || lastModified + expiredInterval < System.currentTimeMillis()) {
                            stop = !lockFile.delete();
                        } else {
                            stop = true;
                        }
                    }
                }
                throw new LockException("failed to lock file: " + file.getPath());
            }
            return this;
        }

        private void watch() {
            do {
                boolean setLastModified = lockFile.setLastModified(System.currentTimeMillis());
                if(!setLastModified) {
                    unlocked.set(true);
                }
                if(!unlocked.get()) {
                    try {
                        Thread.sleep(interval);
                    } catch (InterruptedException e) {
                        unlocked.set(true);
                    }
                }
            } while (!unlocked.get());
            lockFile.delete();
        }

        public FileLock unlock() {
            unlocked.set(true);
            if(thread != null) {
                thread.interrupt();
                do {
                    try {
                        thread.join();
                    } catch(InterruptedException ignored) {

                    }
                } while(thread.isAlive());
                thread = null;
            }
            return this;
        }

        public boolean isUnlocked() {
            return unlocked.get();
        }

        public boolean isLocked() {
            return !isUnlocked();
        }

        @Override public void close() throws Exception {
            unlock();
        }
    }
}
