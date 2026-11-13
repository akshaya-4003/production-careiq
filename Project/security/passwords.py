"""Hash and verify passwords, so that what is stored is a verifier, never a password."""

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError


class PasswordVault(object):
    """Argon2id with a fresh salt per hash."""

    def __init__(self):
        self._hasher = PasswordHasher()

    def hash_password(self, plain_password):
        return self._hasher.hash(plain_password)

    def matches_stored_hash(self, plain_password, stored_hash):
        try:
            return bool(self._hasher.verify(stored_hash, plain_password))
        except (VerifyMismatchError, VerificationError, InvalidHashError, ValueError, TypeError):
            return False

    def needs_rehash(self, stored_hash):
        try:
            return bool(self._hasher.check_needs_rehash(stored_hash))
        except (InvalidHashError, ValueError, TypeError):
            return True
