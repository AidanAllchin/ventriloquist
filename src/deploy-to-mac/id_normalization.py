"""
Standalone phone number and identifier normalization utilities for the on-device deployment.

File: deploy-to-mac/phone_normalization.py
Author: Aidan Allchin
Created: 2025-11-23
"""

import logging
from typing import List, Optional

import phonenumbers

log = logging.getLogger(__name__)


def normalize_phone_number(phone: str, default_region: str = "US") -> Optional[str]:
    """
    Normalize phone number to E.164 format.

    E.164 is the international telephone numbering plan that ensures global uniqueness.
    Format: +[country code][subscriber number] (e.g., +15551234567)

    Args:
        phone: Raw phone number string (e.g., "(555) 123-4567", "555-123-4567", "+1 555 123 4567")
        default_region: Default country code if not specified in the number (default: "US")

    Returns:
        E.164 formatted number (e.g., "+15551234567") or None if invalid

    Examples:
        >>> normalize_phone_number("555-123-4567")
        '+15551234567'
        >>> normalize_phone_number("+1 (555) 123-4567")
        '+15551234567'
        >>> normalize_phone_number("(555) 123-4567", "US")
        '+15551234567'
        >>> normalize_phone_number("+44 20 7123 4567", "GB")
        '+442071234567'
        >>> normalize_phone_number("invalid")
        None
    """
    if not phone or not isinstance(phone, str):
        return None

    # Strip whitespace
    phone = phone.strip()
    if not phone:
        return None

    try:
        # Parse the phone number with the default region
        parsed = phonenumbers.parse(phone, default_region)

        # Validate that it's a possible and valid number
        if phonenumbers.is_valid_number(parsed):
            # Format as E.164 (international format with + prefix)
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        else:
            log.debug(f"Phone number '{phone}' is not valid according to phonenumbers library")
            return None

    except phonenumbers.NumberParseException as e:
        log.debug(f"Could not parse phone number '{phone}': {e}")
        return None
    except Exception as e:
        log.warning(f"Unexpected error normalizing phone number '{phone}': {e}")
        return None

def normalize_identifier(identifier: str, default_region: str = "US") -> Optional[str]:
    """
    Normalize an identifier (phone number or email address).

    This function determines whether the identifier is an email or phone number
    and applies the appropriate normalization:
    - Emails: Converted to lowercase and trimmed
    - Phone numbers: Converted to E.164 format using phonenumbers library

    Args:
        identifier: Phone number or email address
        default_region: Default country code for phone numbers (default: "US")

    Returns:
        Normalized identifier:
        - For emails: lowercase email (e.g., "john@example.com")
        - For phone numbers: E.164 format (e.g., "+15551234567")
        - None if identifier is invalid or cannot be normalized

    Examples:
        >>> normalize_identifier("John@Example.COM")
        'john@example.com'
        >>> normalize_identifier("555-123-4567")
        '+15551234567'
        >>> normalize_identifier("+1 (555) 123-4567")
        '+15551234567'
        >>> normalize_identifier("  user@domain.com  ")
        'user@domain.com'
    """
    if not identifier or not isinstance(identifier, str):
        return None

    # Strip whitespace
    identifier = identifier.strip()
    if not identifier:
        return None

    # Check if it's an email (simple heuristic: contains '@')
    if "@" in identifier:
        # Normalize email: lowercase and trim
        normalized = identifier.lower().strip()
        # Basic validation: must have exactly one @ and something on both sides
        if normalized.count("@") == 1:
            local, domain = normalized.split("@")
            if local and domain:
                return normalized
        log.debug(f"Email identifier '{identifier}' appears malformed")
        return None

    # Otherwise, treat as phone number
    normalized_phone = normalize_phone_number(identifier, default_region)
    if normalized_phone:
        return normalized_phone

    # If normalization failed, return None (don't return as-is)
    # This ensures we only store properly formatted identifiers
    log.debug(f"Could not normalize identifier '{identifier}' as phone or email")
    return None

def normalize_identifier_list(
        identifiers: list[str],
        default_region: str = "US"
    ) -> List[str]:
    """
    Normalize a list of identifiers, filtering out any that cannot be normalized.

    Args:
        identifiers: List of phone numbers and/or email addresses
        default_region: Default country code for phone numbers (default: "US")

    Returns:
        List of normalized identifiers (invalid ones are filtered out)

    Examples:
        >>> normalize_identifier_list(["+1-555-123-4567", "John@Example.COM", "invalid"])
        ['+15551234567', 'john@example.com']
    """
    normalized = []
    for identifier in identifiers:
        norm = normalize_identifier(identifier, default_region)
        if norm:
            normalized.append(norm)
    return normalized
