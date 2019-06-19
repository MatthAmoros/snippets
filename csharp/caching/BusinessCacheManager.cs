 public class BusinessCacheManager
    {
        public const string CACHE_KEY_CUSTOMER_ACCOUNT_ENTRIES = "CAE";
        public const string CACHE_KEY_PROVIDER_ACCOUNT_ENTRIES = "PAE";

        private static Dictionary<string, DateTime> _cacheExpiryDateByKey = new Dictionary<string, DateTime>();
        private TimeSpan _expirySpan = TimeSpan.FromMinutes(10);

        /// <summary>
        /// Return cache or false if cache is expired or not set
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="key"></param>
        /// <param name="cache"></param>
        /// <returns></returns>
        public bool TryGetCachedData<T>(string key, out T cache) where T : new()
        {
            cache = new T();

            try
            {
                cache = (T)MemoryCache.Default[key];
                return IsCacheExpired(key);
            }
            catch(Exception ex)
            {
                return false;
            }            
        }

        /// <summary>
        /// Return cache or false if cache is expired or not set
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="key"></param>
        /// <param name="cache"></param>
        /// <returns></returns>
        public bool SetCachedData<T>(string key, T cache) where T : new()
        {
            try
            {
                MemoryCache.Default[key] = cache;

                if (!_cacheExpiryDateByKey.ContainsKey(key))
                    _cacheExpiryDateByKey.Add(key, DateTime.Now);
                else
                    _cacheExpiryDateByKey[key] = DateTime.Now;

                return true;
            }
            catch (Exception ex)
            {
                return false;
            }
        }

        /// <summary>
        /// Return true if cache expired
        /// </summary>
        /// <param name="key"></param>
        /// <returns></returns>
        private bool IsCacheExpired(string key)
        {
            var lastUpdated = _cacheExpiryDateByKey[key];

            return (DateTime.Compare(lastUpdated, DateTime.Now - _expirySpan) >= 1);
        }
    }
