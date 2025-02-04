const { MongoClient } = require('mongodb');

// Connection URIs
const sourceUri = 'mongodb://127.0.0.1:27017/'; // Source database connection string
const destinationUri = 'mongodb+srv://naibahoadventus:Atlasadven1@billboard.wmuatch.mongodb.net/?retryWrites=true&w=majority&appName=Billboard'; // Destination database connection string

// Database and collection names
const sourceDbName = 'test';
const sourceCollectionName = 'metrics';
const destinationDbName = 'test';
const destinationCollectionName = 'metrics';


async function updateDataEveryThreeSeconds() {
    const sourceClient = new MongoClient(sourceUri);
    const destinationClient = new MongoClient(destinationUri);

    try {
        // Connect to both databases
        await sourceClient.connect();
        await destinationClient.connect();

        const sourceCollection = sourceClient.db(sourceDbName).collection(sourceCollectionName);
        const destinationCollection = destinationClient.db(destinationDbName).collection(destinationCollectionName);

        setInterval(async () => {
            try {
                console.log('Fetching data from source...');
                const data = await sourceCollection.find({}).toArray();

                for (const item of data) {
                    const { title, ...itemWithoutTitle } = item; // Exclude the title field
                    await destinationCollection.updateOne(
                        { _id: itemWithoutTitle._id }, // Match document by _id
                        { $set: itemWithoutTitle }, // Update the document
                        { upsert: true } // Insert if no match is found
                    );
                }

                console.log('Data updated in destination database!');
            } catch (error) {
                console.error('Error during data update:', error);
            }
        }, 5000); // Repeat every 3 seconds

    } catch (error) {
        console.error('Error setting up connections:', error);
    }
}

updateDataEveryThreeSeconds();