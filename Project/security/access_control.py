"""Signed tokens carrying a role, permission checks, and the authority rule.

Every difference between the roles is enforced on the server. Hiding a button in
the dashboard is a courtesy; the service refuses the request on its own account.
"""

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, Header, HTTPException, status

from config import config


class AccessControl(object):
    """Issues and reads the signed tokens."""

    def __init__(self, secret_key=None, algorithm=None, lifetime_minutes=None):
        self.secret_key = config.JWT_SECRET_KEY if secret_key is None else secret_key
        self.algorithm = config.JWT_ALGORITHM if algorithm is None else algorithm
        self.lifetime_minutes = config.TOKEN_LIFETIME_MINUTES if lifetime_minutes is None \
            else lifetime_minutes

    def issue_token(self, user_name, role, display_name=""):
        now = datetime.now(timezone.utc)
        claims = {
            "sub": user_name,
            "role": role,
            "name": display_name or user_name,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=self.lifetime_minutes)).timestamp()),
        }
        return jwt.encode(claims, self.secret_key, algorithm=self.algorithm)

    def read_token(self, authorization_header):
        """Verify the signature and expiry and return the claims; raise ValueError otherwise."""
        if not authorization_header:
            raise ValueError("No token was supplied.")
        token = authorization_header.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        try:
            return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError:
            raise ValueError("The token has expired; sign in again.")
        except jwt.InvalidTokenError as error:
            raise ValueError("The token is not valid: " + str(error))


def permissions_for_role(role):
    """What a role may do; an unknown role holds nothing."""
    return list(config.ROLE_PERMISSIONS.get(role, []))


def role_has_permission(role, permission):
    return permission in permissions_for_role(role)


_GUARD = AccessControl()


def current_user(authorization=Header(default="")):
    """Resolve the caller from the request's bearer token."""
    try:
        claims = _GUARD.read_token(authorization)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(error),
                            headers={"WWW-Authenticate": "Bearer"})
    role = claims.get("role", "")
    if role not in config.ROLE_OPTIONS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="The token carries an unrecognised role " + repr(role) + ".")
    return {
        "user_name": claims.get("sub", ""),
        "display_name": claims.get("name", ""),
        "role": role,
        "permissions": permissions_for_role(role),
    }


def require_permission(permission):
    """A dependency that admits only roles holding `permission`."""

    def check(user=Depends(current_user)):
        if not role_has_permission(user["role"], permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="The " + user["role"] + " role does not hold the '" + permission
                       + "' permission this request needs.")
        return user

    return check


_AUTHORITY_BY_ROLE = {
    config.ROLE_CUSTOMER: 0.0,
    config.ROLE_ASSOCIATE: float(config.ASSOCIATE_AUTHORITY_LIMIT_INR),
    config.ROLE_MANAGER: float(config.MANAGER_AUTHORITY_LIMIT_INR),
    config.ROLE_ADMIN: float(config.HEAD_OF_CARE_AUTHORITY_LIMIT_INR),
}


def authority_limit_for(role):
    """The most a role may commit without escalating, under NRC-CSC-014. Always finite."""
    return float(_AUTHORITY_BY_ROLE.get(role, 0.0))


def lowest_role_that_may_commit(amount_inr):
    """The first role, in ascending order, whose authority covers this amount."""
    for role in config.ROLE_OPTIONS:
        if role != config.ROLE_CUSTOMER and amount_inr <= authority_limit_for(role):
            return role
    return ""


def needs_manager_approval(remedy_type, amount_inr, payment_method=""):
    """Whether a drafted remedy must wait for a person, and why, in words. Rule only."""
    if remedy_type in config.ALWAYS_NEEDS_APPROVAL_REMEDIES:
        return True, ("A {0} always needs a care manager under NRC-CSC-014, whatever the "
                      "amount.".format(remedy_type))
    if remedy_type in config.FULL_REFUND_REMEDIES and payment_method in \
            config.ALWAYS_NEEDS_APPROVAL_PAYMENT_METHODS:
        return True, ("A full refund on an order paid by {0} always needs a care manager under "
                      "NRC-CSC-014.".format(payment_method))
    if float(amount_inr) > config.ASSOCIATE_AUTHORITY_LIMIT_INR:
        return True, ("INR {0:,.2f} is above the associate authority limit of INR {1:,.2f} "
                      "under NRC-CSC-014.".format(float(amount_inr),
                                                  config.ASSOCIATE_AUTHORITY_LIMIT_INR))
    return False, ("INR {0:,.2f} is within the associate authority limit of INR {1:,.2f}."
                   .format(float(amount_inr), config.ASSOCIATE_AUTHORITY_LIMIT_INR))
