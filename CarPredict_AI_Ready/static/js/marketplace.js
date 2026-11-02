(() => {
    'use strict';

    const listingGrid = document.querySelector('#listingGrid');
    const listingStatus = document.querySelector('#listingStatus');
    const emptyState = document.querySelector('#emptyState');
    const search = document.querySelector('#listingSearch');
    const form = document.querySelector('#listingForm');
    const submitButton = document.querySelector('#submitListing');
    const errorMessage = document.querySelector('#listingError');
    const successMessage = document.querySelector('#listingSuccess');
    let listings = [];

    document.querySelector('#theme').addEventListener('click', () => {
        document.body.classList.toggle('dark');
    });

    const formatPrice = value => new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        maximumFractionDigits: 0
    }).format(value);

    function addText(parent, tag, className, value) {
        const element = document.createElement(tag);
        if (className) element.className = className;
        element.textContent = value;
        parent.appendChild(element);
        return element;
    }

    function createListingCard(listing) {
        const card = document.createElement('article');
        card.className = 'listing-card';
        const topline = document.createElement('div');
        topline.className = 'listing-topline';
        addText(topline, 'span', 'listing-tag', 'For sale');
        if (listing.created_at) {
            const posted = new Date(`${listing.created_at.replace(' ', 'T')}Z`);
            if (!Number.isNaN(posted.getTime())) {
                addText(topline, 'time', 'listing-date', `Listed ${new Intl.DateTimeFormat(
                    undefined,
                    {month: 'short', day: 'numeric'}
                ).format(posted)}`);
            }
        }
        card.appendChild(topline);
        addText(card, 'h3', '', `${listing.year} ${listing.make} ${listing.model}`);
        addText(card, 'span', 'listing-location', listing.city);
        addText(card, 'strong', 'listing-price', formatPrice(listing.price_rupees));

        const meta = document.createElement('ul');
        meta.className = 'listing-meta';
        [
            `${new Intl.NumberFormat('en-IN').format(listing.kilometers)} km driven`,
            listing.fuel_type,
            listing.transmission,
            listing.condition
        ].filter(Boolean).forEach(value => addText(meta, 'li', '', value));
        card.appendChild(meta);

        if (listing.description) {
            addText(card, 'p', 'listing-description', listing.description);
        }

        const seller = document.createElement('div');
        seller.className = 'seller-info';
        addText(seller, 'strong', '', `Listed by ${listing.seller_name}`);
        const actions = document.createElement('div');
        actions.className = 'seller-actions';

        if (listing.seller_phone) {
            const phone = document.createElement('a');
            phone.href = `tel:${listing.seller_phone.replace(/[^\d+]/g, '')}`;
            phone.textContent = 'Call seller';
            actions.appendChild(phone);
        }
        if (listing.seller_email) {
            const email = document.createElement('a');
            email.href = `mailto:${listing.seller_email}`;
            email.textContent = 'Email seller';
            actions.appendChild(email);
        }
        seller.appendChild(actions);
        card.appendChild(seller);
        return card;
    }

    function renderListings() {
        const query = search.value.trim().toLocaleLowerCase();
        const filtered = listings.filter(listing =>
            [listing.make, listing.model, listing.city]
                .some(value => (value || '').toLocaleLowerCase().includes(query))
        );
        listingGrid.replaceChildren(...filtered.map(createListingCard));
        emptyState.hidden = filtered.length > 0;
        listingStatus.textContent = listings.length
            ? `${filtered.length} of ${listings.length} car${listings.length === 1 ? '' : 's'} listed`
            : '';
        if (filtered.length === 0) {
            emptyState.querySelector('h3').textContent = listings.length
                ? 'No matching cars'
                : 'No cars listed just yet';
            emptyState.querySelector('p').textContent = listings.length
                ? 'Try another make, model, or city.'
                : 'When someone lists a car, you’ll find it here.';
            if (listings.length > 0) {
                listingStatus.textContent = '';
            }
        }
    }

    async function loadListings() {
        listingStatus.textContent = 'Loading listings…';
        try {
            const response = await fetch('/api/listings');
            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.error || 'Could not load car listings.');
            }
            listings = data.listings;
            renderListings();
        } catch (error) {
            listingStatus.textContent = error.message;
        }
    }

    search.addEventListener('input', renderListings);

    form.addEventListener('submit', async event => {
        event.preventDefault();
        errorMessage.hidden = true;
        successMessage.hidden = true;
        submitButton.disabled = true;
        submitButton.textContent = 'Publishing…';

        const payload = Object.fromEntries(new FormData(form).entries());
        for (const field of ['year', 'kilometers', 'price_rupees']) {
            payload[field] = Number(payload[field]);
        }

        try {
            const response = await fetch('/api/listings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.error || 'Could not publish your listing.');
            }
            listings.unshift(data.listing);
            renderListings();
            form.reset();
            successMessage.textContent = 'Your car listing is now published.';
            successMessage.hidden = false;
            document.querySelector('#listings').scrollIntoView({behavior: 'smooth'});
        } catch (error) {
            errorMessage.textContent = error.message;
            errorMessage.hidden = false;
        } finally {
            submitButton.disabled = false;
            submitButton.textContent = 'Publish listing';
        }
    });

    loadListings();
})();
