function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

// створюємо "живий вивід" на сторінці
const box = document.createElement("textarea");
box.style = `
    position: fixed;
    top: 10px;
    right: 10px;
    width: 500px;
    height: 600px;
    z-index: 999999;
    background: #111;
    color: #0f0;
    font-size: 11px;
`;
document.body.appendChild(box);

const header = [
    "listing_id",
    "real_mls",
    "price",

    "property_type",
    "style",
    "year_built",

    "street",
    "district",
    "city",


    "floor_area_sqft",
    "lot_size_sqft",
    "lot_frontage_ft",

    "rooms",
    "bedrooms",
    "bathrooms",

    "parking_spaces",
    "parking_features",

    "cooling_features",
    "fireplace_features",

    "interior_features",
    "appliances",

    "view",

    "property_taxes",
    "tax_year",

    "walkscore",


    "agency",

    "virtual_tour",

    "photo_count",

    "summary_url"
];


function findFeature(item, name) {
    const row = [...item.querySelectorAll(".carac-container")]
        .find(x =>
            x.querySelector(".carac-title")
                ?.textContent.trim() === name
        );

    return row
        ?.querySelector(".carac-value")
        ?.textContent.trim() || "";
}
let csvLines = [];
csvLines.push(header.join(","));

function appendCSV(rows) {
    const escape = v => `"${String(v ?? "").replace(/"/g, '""')}"`;

    for (const r of rows) {
        csvLines.push(r.map(escape).join(","));
    }

    box.value = csvLines.join("\n");
}

async function fetchPage(page = 1) {

	const payload = {
		mode: "Result",
		searchView: "Summary",
		sortSeed: 1632068101,
		sort: "None",
		pageSize: 1,
		page,
		query: {
			SearchName: "",
			UseGeographyShapes: 0,
			Filters: [],
			FieldsValues: [
				{
					fieldId: "Category",
					value: "Residential",
					fieldConditionId: "",
					valueConditionId: ""
				},
				{
					fieldId: "SellingType",
					value: "Sale",
					fieldConditionId: "",
					valueConditionId: ""
				},
				{
					fieldId: "PropertyArea",
					value: "SquareFeet",
					fieldConditionId: "IsNotLot",
					valueConditionId: ""
				},
				{
					fieldId: "LandArea",
					value: "SquareFeet",
					fieldConditionId: "",
					valueConditionId: ""
				},
				{
					fieldId: "SalePrice",
					value: 0,
					fieldConditionId: "ForSale",
					valueConditionId: ""
				},
				{
					fieldId: "SalePrice",
					value: 999999999999,
					fieldConditionId: "ForSale",
					valueConditionId: ""
				}
			],
			BrokerCode: null,
			OfficeKey: null
		},
		region: null
	};

    const res = await fetch("https://realtylink.org/Property/GetInscriptions", {
        method: "POST",
        headers: {
            "Content-Type": "application/json;charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest"
        },
        body: JSON.stringify(payload)
    });

    const data = await res.json();
    return data.d.Result.html;
}

function parse(html) {
    const doc = new DOMParser().parseFromString(html, "text/html");

    const item = doc.querySelector(".property-summary-item");

    if (!item) return [];

    const price =
        item.querySelector("meta[itemprop='price']")?.content || "";

    const type =
        item.querySelector("h1 span[data-id='PageTitle']")
            ?.textContent.trim() || "";

    const address =
        item.querySelector("h2[itemprop='address']")
            ?.textContent.trim() || "";

    const addressParts = address.split(",");

    const street = addressParts[0]?.trim() || "";
    const district = addressParts[1]?.trim() || "";
    const city = addressParts[2]?.trim() || "";

    const sqft =
        [...item.querySelectorAll(".carac-container")]
            .find(x =>
                x.querySelector(".carac-title")?.textContent.includes("Floor Area")
            )
            ?.querySelector(".carac-value")
            ?.textContent.replace(/[^\d]/g, "") || "";

    const beds =
        item.querySelector(".cac")
            ?.textContent.match(/\d+/)?.[0] || "";

    const baths =
        item.querySelector(".sdb")
            ?.textContent.match(/\d+/)?.[0] || "";

    const photos =
        item.querySelector(".photo-btn:not(.virtual-tour)")
            ?.textContent.match(/\d+/)?.[0] || "";

    const listingId =
        item.querySelector("#ListingId")
            ?.textContent.trim() || "";

    const realMls =
        item.querySelector("#ListingDisplayId")
            ?.textContent.trim() || "";

    const agency =
        item.querySelector(".broker-info__agency-name")
            ?.textContent.trim() || "";


    const summaryUrl =
        item.querySelector("#SummaryUrl")
            ?.textContent.trim() || "";

    const url = summaryUrl
        ? `https://realtylink.org${summaryUrl}`
        : "";

    const badges = [
        ...item.querySelectorAll(".badges-container .badge")
    ]
        .map(x => x.textContent.trim())
        .filter(Boolean)
        .join(" | ");

const style = findFeature(item, "Style");
const yearBuilt = findFeature(item, "Year Built");
const lotSize = findFeature(item, "Lot Size");
const lotFrontage = findFeature(item, "Lot Frontage");
const appliances = findFeature(item, "Appliances");
const cooling = findFeature(item, "Cooling Features");
const fireplace = findFeature(item, "Fireplace Features");
const parkingFeatures = findFeature(item, "Parking Features");
const parkingSpaces = findFeature(item, "Parking Spaces");
const interiorFeatures = findFeature(item, "Interior Features");
const view = findFeature(item, "View");
const propertyTaxes = findFeature(item, "Property Taxes");

const rooms =
    item.querySelector(".piece")
        ?.textContent.match(/\d+/)?.[0] || "";

const walkscore =
    item.querySelector(".walkscore span")
        ?.textContent.trim() || "";



const virtualTour =
    item.querySelector(".virtual-tour")
        ?.getAttribute("onclick")
        ?.match(/window\.open\('([^']+)'/)?.[1] || "";

const taxYear =
    propertyTaxes.match(/\((\d{4})\)/)?.[1] || "";

    return [[
    listingId,
    realMls,
    price,

    type,
    style,
    yearBuilt,

    street,
    district,
    city,


    sqft,
    lotSize.replace(/[^\d]/g, ""),
    lotFrontage,

    rooms,
    beds,
    baths,

    parkingSpaces,
    parkingFeatures,

    cooling,
    fireplace,

    interiorFeatures,
    appliances,

    view,

    propertyTaxes,
    taxYear,

    walkscore,


    agency,



    virtualTour,

    photos,

    url
]];
}

function randomDelay() {
    return Math.floor(Math.random() * (30000 - 3000 + 1)) + 3000;
}

async function run() {
	const pageCount = 5000;
	const startPage = 1;
    for (let page = startPage; page <= pageCount ; page++) {
        console.log("Page", page);

        const html = await fetchPage(page);
        const rows = parse(html);
		if (!rows.length) {
			console.log("No data found. Stop.");
			break;
		}

        appendCSV(rows); // 👈 ОНОВЛЕННЯ CSV ОДРАЗУ

        console.log("Added:", rows.length);

        if (page < pageCount ) {
			const delay = randomDelay();
			console.log(`Waiting ${Math.round(delay / 1000)}s...`);
			await sleep(delay);
		}
    }

    console.log("DONE ✅");
    console.log(box.value);
}

run();