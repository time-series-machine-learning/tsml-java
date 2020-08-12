package utilities;

import java.io.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class FileUtils {


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
