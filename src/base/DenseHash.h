#pragma once

#include <vector>
#include <cassert>
#include <utility>
#include <functional>

// Internal implementation of DenseHashSet and DenseHashMap
namespace detail
{
    template <typename Key, typename Item, typename Hash, typename Eq>
    class DenseHashTable
    {
    public:
        typedef typename std::vector<Item>::const_iterator const_iterator;

        DenseHashTable(size_t capacity, const Hash& hash, const Eq& eq)
            : filled(0)
            , hash(hash)
            , eq(eq)
        {
            if (capacity)
            {
                items.reserve(capacity);
                rehash(capacity);
            }
        }

        const_iterator begin() const
        {
            return items.begin();
        }

        const_iterator end() const
        {
            return items.end();
        }

        bool empty() const
        {
            return items.empty();
        }

        size_t size() const
        {
            return items.size();
        }

        size_t bucket_count() const
        {
            return buckets.size();
        }

        float load_factor() const
        {
            return buckets.empty() ? 0 : float(filled) / float(buckets.size());
        }

        void clear()
        {
            items.clear();
            buckets.clear();
            filled = 0;
        }

    protected:
        std::vector<Item> items;
        std::vector<int32_t> buckets;
        size_t filled; // number of non-empty buckets

        Hash hash;
        Eq eq;

        void rehash()
        {
            if (filled >= buckets.size() * 3 / 4)
            {
                rehash(items.size() * 2);
            }
        }

        void rehash(size_t capacity)
        {
            size_t newbuckets = 16;
            while (newbuckets < capacity) newbuckets *= 2;

            size_t hashmod = newbuckets - 1;

            std::vector<int32_t>(newbuckets, -1).swap(buckets);

            for (size_t i = 0; i < items.size(); ++i)
            {
                size_t bucket = hash(getKey(items[i])) & hashmod;

                for (size_t probe = 0; probe <= hashmod; ++probe)
                {
                    if (buckets[bucket] < 0)
                    {
                        buckets[bucket] = i;
                        break;
                    }

                    // Hash collision, quadratic probing
                    bucket = (bucket + probe + 1) & hashmod;
                }
            }

            filled = items.size();
        }

        int find_bucket(const Key& key) const
        {
            if (buckets.empty())
                return -1;

            size_t hashmod = buckets.size() - 1;
            size_t bucket = hash(key) & hashmod;

            for (size_t probe = 0; probe <= hashmod; ++probe)
            {
                int32_t probe_index = buckets[bucket];

                // Element does not exist, insert here
                if (probe_index == -1)
                    return -1;

                // Not a tombstone and key matches
                if (probe_index >= 0 && eq(getKey(items[probe_index]), key))
                    return bucket;

                // Hash collision, quadratic probing
                bucket = (bucket + probe + 1) & hashmod;
            }

            // Hash table is full - this should not happen
            assert(false);
            return -1;
        }

        std::pair<Item*, bool> insert_item(const Key& key)
        {
            assert(!buckets.empty());

            size_t hashmod = buckets.size() - 1;
            size_t bucket = hash(key) & hashmod;

            for (size_t probe = 0; probe <= hashmod; ++probe)
            {
                int32_t probe_index = buckets[bucket];

                // Element does not exist or a tombstone, insert here
                if (probe_index < 0)
                {
                    buckets[bucket] = items.size();
                    filled += probe_index == -1;

                    items.push_back(Item());
                    getKey(items.back()) = key;

                    return std::make_pair(&items.back(), true);
                }

                // Key matches, insert here
                if (eq(getKey(items[probe_index]), key))
                    return std::make_pair(&items[probe_index], false);

                // Hash collision, quadratic probing
                bucket = (bucket + probe + 1) & hashmod;
            }

            // Hash table is full - this should not happen
            assert(false);
            return std::make_pair(static_cast<Item*>(0), false);
        }

        void erase_bucket(int bucket)
        {
            assert(bucket >= 0);

            int32_t probe_index = buckets[bucket];
            assert(probe_index >= 0);

            // move last key
            int probe_bucket = find_bucket(getKey(items.back()));
            assert(probe_bucket >= 0);
            assert(buckets[probe_bucket] == int32_t(items.size() - 1));

            items[probe_index] = items.back();
            buckets[probe_bucket] = probe_index;

            items.pop_back();

            // mark bucket as tombstone
            buckets[bucket] = -2;
        }

    private:
        // Interface to support both key and pair<key, value>
        static const Key& getKey(const Key& item) { return item; }
        static Key& getKey(Key& item) { return item; }
        template <typename Value> static const Key& getKey(const std::pair<Key, Value>& item) { return item.first; }
        template <typename Value> static Key& getKey(std::pair<Key, Value>& item) { return item.first; }
    };
}

template <typename Key, typename Hash = std::hash<Key>, typename Eq = std::equal_to<Key>>
class DenseHashSet: public detail::DenseHashTable<Key, Key, Hash, Eq>
{
public:
    explicit DenseHashSet(size_t capacity = 0, const Hash& hash = Hash(), const Eq& eq = Eq())
        : detail::DenseHashTable<Key, Key, Hash, Eq>(capacity, hash, eq)
    {
    }

    bool contains(const Key& key) const
    {
        return this->find_bucket(key) >= 0;
    }

    bool insert(const Key& key)
    {
        this->rehash();

        return this->insert_item(key).second;
    }

    void erase(const Key& key)
    {
        int bucket = this->find_bucket(key);

        if (bucket >= 0)
            this->erase_bucket(bucket);
    }
};

// This is a faster alternative of std::unordered_map, but it does not implement the same interface (i.e. it does not support erasing and has contains() instead of find())
template <typename Key, typename Value, typename Hash = std::hash<Key>, typename Eq = std::equal_to<Key>>
class DenseHashMap: public detail::DenseHashTable<Key, std::pair<Key, Value>, Hash, Eq>
{
public:
    explicit DenseHashMap(size_t capacity = 0, const Hash& hash = Hash(), const Eq& eq = Eq())
        : detail::DenseHashTable<Key, std::pair<Key, Value>, Hash, Eq>(capacity, hash, eq)
    {
    }

    const Value* find(const Key& key) const
    {
        int bucket = this->find_bucket(key);

        return bucket < 0 ? NULL : &this->items[this->buckets[bucket]].second;
    }

    Value& operator[](const Key& key)
    {
        this->rehash();

        return this->insert_item(key).first->second;
    }

    void erase(const Key& key)
    {
        int bucket = this->find_bucket(key);

        if (bucket >= 0)
            this->erase_bucket(bucket);
    }
};
